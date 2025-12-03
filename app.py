from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from urllib.parse import urlparse

import pickle
import traceback
import numpy as np
import asyncio
import asyncpraw
from apify_client import ApifyClient
from playwright.sync_api import sync_playwright
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
from dotenv import load_dotenv
import os
import json
from fpdf import FPDF  # Requires 'pip install fpdf2'
from flask import send_file
import io
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# --- Initial NLTK VADER Download ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER lexicon for sentiment analysis...")
    nltk.download('vader')
    print("Download complete.")

# ---------- Config ----------
MODEL_PATH = './content/model_nb.pkl'

load_dotenv()

REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
APIFY_API_TOKEN = os.getenv('APIFY_API_TOKEN')

# Thresholds
THRESHOLD_ABNORMAL = 0.75

# Platform configuration (single source of truth)
PLATFORMS = {
    'reddit': {
        'name': 'Reddit',
        'icon': 'fab fa-reddit-alien',
        'placeholder': 'Enter Reddit username (e.g., popular_user_name)'
    },
    'instagram': {
        'name': 'Instagram',
        'icon': 'fab fa-instagram',
        'placeholder': 'Enter Instagram username or profile URL'
    },
    'twitter': {
        'name': 'Twitter/X',
        'icon': 'fab fa-twitter',
        'placeholder': 'Enter Twitter/X username'
    }
}

app = Flask(__name__)
app.secret_key = os.urandom(24)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mindspace.db'
db = SQLAlchemy(app)

# --- Flask-Login Configuration ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "Please log in to access this page and save your history."


@login_manager.user_loader
def load_user(user_id):
    # Using SQLAlchemy 1.x query syntax for compatibility
    return User.query.get(int(user_id))


# --- Database Models ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    history = db.relationship('AnalysisHistory', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class AnalysisHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date_generated = db.Column(db.DateTime, default=datetime.utcnow)
    platform = db.Column(db.String(50), nullable=False)
    username_analyzed = db.Column(db.String(100), nullable=False)
    verdict = db.Column(db.String(50), nullable=False)
    avg_abnormal_prob = db.Column(db.Float, nullable=False)
    full_results_json = db.Column(db.Text, nullable=False)


# Create tables
with app.app_context():
    db.create_all()

# ---------- Load Model & Sentiment Analyzer ----------
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}. Prediction will fail.")
    model = None

sia = SentimentIntensityAnalyzer()


# ---------- Helpers ----------
def get_class_indices():
    """Gets the index for the 'abnormal' class from the model."""
    if model and hasattr(model, "classes_"):
        classes = list(model.classes_)
        try:
            return classes.index(1)
        except ValueError:
            return 1 if len(classes) > 1 else 0
    return 1


IDX_ABNORMAL = get_class_indices()


def analyze_posts(posts):
    """
    Analyzes a list of post objects.
    """
    if not posts:
        return None

    analyzed_posts = []
    total_abnormal_prob = 0
    total_neg_sentiment = 0
    total_pos_sentiment = 0
    total_neu_sentiment = 0

    model_available = model is not None

    for post in posts:
        text = post.get('text', '')
        if not text:
            continue
        prob_abnormal = 0.0
        if model_available:
            try:
                proba = model.predict_proba([text])[0]
                prob_abnormal = float(proba[IDX_ABNORMAL])
            except Exception:
                prob_abnormal = 0.0

        sentiment = sia.polarity_scores(text)
        analyzed_posts.append({
            'text': text,
            'prob_abnormal': prob_abnormal,
            'sentiment': sentiment,
            'date': datetime.fromtimestamp(post.get('timestamp')).strftime('%Y-%m-%d'),
            'timestamp': post.get('timestamp')
        })

        total_abnormal_prob += prob_abnormal
        total_neg_sentiment += sentiment['neg']
        total_pos_sentiment += sentiment['pos']
        total_neu_sentiment += sentiment['neu']

    analyzed_posts.sort(key=lambda x: x['timestamp'])

    graph_labels = [p['date'] for p in analyzed_posts]
    graph_abnormal_prob = [p['prob_abnormal'] for p in analyzed_posts]
    graph_neg_sentiment = [p['sentiment']['neg'] for p in analyzed_posts]

    post_count = len(analyzed_posts)
    aggregates = {
        'avg_abnormal_prob': total_abnormal_prob / post_count if post_count > 0 else 0,
        'avg_neg_sentiment': total_neg_sentiment / post_count if post_count > 0 else 0,
        'avg_pos_sentiment': total_pos_sentiment / post_count if post_count > 0 else 0,
        'avg_neu_sentiment': total_neu_sentiment / post_count if post_count > 0 else 0,
        'post_count': post_count
    }

    verdict = "High Concern" if aggregates['avg_abnormal_prob'] >= THRESHOLD_ABNORMAL else "Low Concern"

    return {
        'verdict': verdict,
        'posts': analyzed_posts,
        'aggregates': aggregates,
        'graph_data': {
            'labels': graph_labels,
            'abnormal_prob': graph_abnormal_prob,
            'neg_sentiment': graph_neg_sentiment
        },
        'threshold': THRESHOLD_ABNORMAL
    }


# --- Scraping Functions (Real Logic Enabled) ---

async def fetch_user_comments_async(username, limit=25):
    posts = []
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        raise RuntimeError("Reddit API credentials (CLIENT_ID, SECRET, USER_AGENT) are not set. Cannot fetch data.")

    async with asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    ) as reddit:
        try:
            user = await reddit.redditor(username)
            await user.load()

        except Exception:
            raise ValueError(f"Reddit user '{username}' not found or API error.")

        async for comment in user.comments.new(limit=limit):
            if comment.created_utc and comment.body:
                posts.append({'text': comment.body, 'timestamp': comment.created_utc})
    return posts


def fetch_user_comments(username, limit=25):
    return asyncio.run(fetch_user_comments_async(username, limit))

def fetch_instagram_captions(profile_input, limit=10):
    if not APIFY_API_TOKEN:
        raise RuntimeError("Apify API token is not set.")
    
    profile_input = profile_input.strip()
    if not profile_input:
        raise ValueError("Please provide an Instagram username or profile URL.")
    
    profile_url = (
        f"https://www.instagram.com/{profile_input}/"
        if not profile_input.startswith("http")
        else profile_input
    )
    
    client = ApifyClient(APIFY_API_TOKEN)
    run_input = {
        "directUrls": [profile_url],
        "resultsType": "posts",
        "resultsLimit": limit,
    }
    run = client.actor("apify/instagram-scraper").call(run_input=run_input)
    
    posts = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        caption = item.get("caption")
        ts = item.get("timestamp")
        if not caption or not ts:
            continue
        try:
            timestamp = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
        except ValueError:
            continue
        posts.append({'text': caption, 'timestamp': timestamp})
    
    return posts


def fetch_twitter_tweets(username, limit=20):
    posts = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"https://twitter.com/{username}", timeout=60000)

            if "Sorry, that page doesn’t exist!" in page.content():
                browser.close()
                raise ValueError(f"Twitter user '{username}' not found.")

            page.wait_for_selector("article[role='article']", timeout=10000)

            seen_tweets = set()
            scroll_count = 0
            max_scrolls = 10

            while len(posts) < limit and scroll_count < max_scrolls:
                articles = page.query_selector_all("article[role='article']")

                for article in articles:
                    try:
                        text_elem = article.query_selector("div[lang]")
                        time_elem = article.query_selector("time")

                        if text_elem and time_elem:
                            text = text_elem.inner_text().strip()
                            if text and text not in seen_tweets:
                                datetime_str = time_elem.get_attribute("datetime")
                                timestamp = datetime.fromisoformat(datetime_str.replace('Z', '+00:00')).timestamp()

                                posts.append({'text': text, 'timestamp': timestamp})
                                seen_tweets.add(text)

                                if len(posts) >= limit:
                                    break
                    except Exception:
                        continue

                if len(posts) >= limit:
                    break

                page.mouse.wheel(0, 3000)
                page.wait_for_timeout(2000)

                scroll_count += 1

            browser.close()
    except Exception as e:
        raise RuntimeError(f"Twitter/X Scraping Error: {e}")

    return posts


# ---------- Authentication Routes ----------

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user:
            flash('Username already exists. Please choose a different one.', 'danger')
        else:
            new_user = User(username=username)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))


# ---------- Core Routes ----------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/how_it_works')
def how_it_works():
    return render_template('how_it_works.html')


@app.route('/faq')
def faq():
    return render_template('faq.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


@app.route('/terms')
def terms():
    return render_template('terms.html')


@app.route('/risk_info')
def risk_info():
    return render_template('risk_info.html')


@app.route('/history')
@login_required
def history():
    user_history = AnalysisHistory.query.filter_by(user_id=current_user.id).order_by(
        AnalysisHistory.date_generated.desc()
    ).all()

    history_data = []
    for record in user_history:
        try:
            full_results = json.loads(record.full_results_json)
        except json.JSONDecodeError:
            full_results = {}

        history_data.append({
            'id': record.id,
            'date': record.date_generated.strftime("%Y-%m-%d %H:%M:%S"),
            'platform': record.platform,
            'user_analyzed': record.username_analyzed,
            'verdict': record.verdict,
            'avg_prob': record.avg_abnormal_prob,
            'full_results': full_results
        })

    return render_template('history.html', history_data=history_data)


# --- Analysis Input Route ---

@app.route('/analyze/<platform>', methods=['GET'])
def input_form(platform):
    platform = platform.lower()
    if platform not in PLATFORMS:
        return redirect(url_for('home'))

    return render_template(
        'input_form.html',
        platform=platform,
        details=PLATFORMS[platform],
        error=None
    )


# --- Unified Analysis Route (POST) ---

@app.route('/results', methods=['POST'])
def run_analysis():
    platform = request.form.get('platform', '').lower()
    username = request.form.get('username', '').strip()

    if platform not in PLATFORMS:
        return redirect(url_for('home'))

    if not username:
        return render_template(
            'input_form.html',
            error="⚠️ Please provide a username/profile for analysis.",
            platform=platform,
            details=PLATFORMS[platform]
        )

    try:
        posts = []
        source_name = PLATFORMS[platform]['name']

        # 1. Fetching/Scraping (REAL DATA)
        if platform == 'reddit':
            posts = fetch_user_comments(username, limit=25)
        elif platform == 'instagram':
            posts = fetch_instagram_captions(username, limit=12)
        elif platform == 'twitter':
            posts = fetch_twitter_tweets(username, limit=20)

        if not posts:
            return render_template(
                'input_form.html',
                error=f"⚠️ No recent public posts found for '{username}' on {source_name}. "
                      f"Ensure the profile is public and active, and check your API keys.",
                platform=platform,
                details=PLATFORMS[platform]
            )

        # 2. Analysis
        results = analyze_posts(posts)

        final_data = {
            'results': results,
            'source': source_name,
            'user': username,
            'date_generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # --- History Storage ---
        if current_user.is_authenticated:
            new_history = AnalysisHistory(
                user_id=current_user.id,
                platform=source_name,
                username_analyzed=username,
                verdict=results['verdict'],
                avg_abnormal_prob=results['aggregates']['avg_abnormal_prob'],
                full_results_json=json.dumps(final_data)
            )
            db.session.add(new_history)
            db.session.commit()
            flash(f"Analysis for {username} saved to history!", 'success')
        else:
            flash("Log in to save this analysis to your history.", 'info')
        # --- End History Storage ---

        session['last_analysis_data'] = final_data
        return render_template('results.html', **final_data)

    except (RuntimeError, ValueError) as e:
        return render_template(
            'input_form.html',
            error=str(e),
            platform=platform,
            details=PLATFORMS[platform]
        )
    except Exception as e:
        traceback.print_exc()
        return render_template(
            'input_form.html',
            error=f"⚠️ An internal server error occurred: {e}",
            platform=platform,
            details=PLATFORMS[platform]
        )
        
def _sanitize_text_basic(text, max_len=None):
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\n", " ")
    if max_len is not None:
        text = text[:max_len]
    text = "".join(ch if ord(ch) < 256 else "?" for ch in text)
    return text


def _soft_wrap_long_words(text, max_word_len=40):
    parts = text.split(" ")
    out = []
    for word in parts:
        if len(word) <= max_word_len:
            out.append(word)
        else:
            start = 0
            while start < len(word):
                out.append(word[start:start + max_word_len])
                start += max_word_len
    return " ".join(out)

def create_abnormality_chart(graph_data):
    labels = graph_data.get('labels', [])
    probs = graph_data.get('abnormal_prob', [])

    if not labels or not probs:
        return None

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(labels, probs, marker='o')
    ax.set_title('Abnormal Probability Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', linewidth=0.5)
    fig.autofmt_xdate(rotation=45)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='PNG')
    plt.close(fig)
    buf.seek(0)
    return buf


def create_sentiment_chart(aggregates):
    neg = aggregates.get('avg_neg_sentiment', 0)
    pos = aggregates.get('avg_pos_sentiment', 0)
    neu = aggregates.get('avg_neu_sentiment', 0)

    fig, ax = plt.subplots(figsize=(4, 3))
    labels = ['Negative', 'Positive', 'Neutral']
    values = [neg, pos, neu]

    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_title('Average Sentiment')
    ax.grid(axis='y', linestyle='--', linewidth=0.5)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='PNG')
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(data, title="MindSpace Social Analysis Report"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)

    def sanitize(text, max_len=None):
        if not isinstance(text, str):
            text = str(text)
        text = text.replace("\n", " ")
        if max_len is not None:
            text = text[:max_len]
        text = "".join(ch if ord(ch) < 256 else "?" for ch in text)
        return text

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin

    # ------------- PAGE 1: TITLE + SUMMARY + GRAPHS -------------

    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(220, 20, 60)
    pdf.cell(0, 10, txt=title, ln=1, align="C")
    pdf.ln(3)

    # Header info
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.set_fill_color(240, 240, 240)
    header_text = (
        f"User: {data['user']}  |  "
        f"Source: {data['source']}  |  "
        f"Date: {data['date_generated']}"
    )
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(usable_width, 8, sanitize(header_text), border=0, align="C", fill=True)
    pdf.ln(4)

    # Separator line
    pdf.set_draw_color(200, 200, 200)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(6)

    # Section header helper
    def section_header(text):
        pdf.set_font("Arial", "B", 13)
        pdf.set_text_color(255, 255, 255)
        pdf.set_fill_color(30, 30, 60)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_width, 8, text, border=0, align="L", fill=True)
        pdf.ln(3)
        pdf.set_text_color(0, 0, 0)

    agg = data["results"]["aggregates"]
    verdict = data["results"]["verdict"]

    # ---- Summary ----
    section_header("Summary Analysis")

    pdf.set_font("Arial", "", 11)

    if verdict == "High Concern":
        pdf.set_text_color(200, 0, 0)
    else:
        pdf.set_text_color(0, 150, 0)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 7, f"Overall Verdict: {verdict}", ln=1)
    pdf.set_text_color(0, 0, 0)

    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 6, f"Average Abnormal Probability: {agg['avg_abnormal_prob']:.2f}", ln=1)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 6, f"Average Negative Sentiment: {agg['avg_neg_sentiment']:.2f}", ln=1)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 6, f"Average Positive Sentiment: {agg['avg_pos_sentiment']:.2f}", ln=1)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 6, f"Average Neutral Sentiment: {agg['avg_neu_sentiment']:.2f}", ln=1)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 6, f"Total Posts Analyzed: {agg['post_count']}", ln=1)

    pdf.ln(6)

    # ---- Graphs on the same page ----
    section_header("Graphs & Trends")

    graph_data = data['results'].get('graph_data', {})
    aggregates = data['results'].get('aggregates', {})

    abnormal_buf = create_abnormality_chart(graph_data)
    sentiment_buf = create_sentiment_chart(aggregates)

    # First chart: full width, moderate height
    if abnormal_buf:
        pdf.set_font("Arial", "B", 12)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 7, "Abnormal Probability Over Time", ln=1)
        pdf.ln(2)

        w1 = usable_width
        h1 = 60  # mm height
        pdf.image(abnormal_buf, x=pdf.l_margin, y=pdf.get_y(), w=w1, h=h1)
        pdf.ln(h1 + 5)

    # Second chart: slightly smaller width under the first
    if sentiment_buf:
        pdf.set_font("Arial", "B", 12)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 7, "Average Sentiment", ln=1)
        pdf.ln(2)

        w2 = usable_width * 0.75
        h2 = 45
        pdf.image(sentiment_buf, x=pdf.l_margin, y=pdf.get_y(), w=w2, h=h2)
        pdf.ln(h2 + 5)

    # ------------- PAGE 2+: DETAILED POSTS -------------

    pdf.add_page()
    section_header("Detailed Post Analysis (Top 10 Posts)")

    posts = data["results"].get("posts", [])[:10]

    if not posts:
        pdf.set_font("Arial", "I", 11)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_width, 6, "No posts available for detailed analysis.")
    else:
        for i, post in enumerate(posts, start=1):
            pdf.set_font("Arial", "B", 12)
            pdf.set_text_color(30, 144, 255)
            pdf.set_x(pdf.l_margin)
            pdf.cell(0, 7, f"Post {i}", ln=1)
            pdf.set_text_color(0, 0, 0)

            pdf.set_font("Arial", "", 10)
            pdf.set_fill_color(245, 245, 245)

            text = sanitize(post.get("text", ""), max_len=200)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(usable_width, 6, f"\"{text}...\"", border=0, fill=True)
            pdf.ln(1)

            pdf.set_font("Arial", "", 9)
            pdf.set_text_color(80, 80, 80)

            stats = (
                f"Neg: {post['sentiment']['neg']:.2f}  |  "
                f"Pos: {post['sentiment']['pos']:.2f}  |  "
                f"Neu: {post['sentiment']['neu']:.2f}  |  "
                f"Abnormal Prob: {post['prob_abnormal']:.2f}  |  "
                f"Date: {post['date']}"
            )
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(usable_width, 5, sanitize(stats), border=0)
            pdf.ln(4)
            pdf.set_text_color(0, 0, 0)

    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer


# --- PDF Download Route for LIVE Analysis ---
@app.route('/download_report', methods=['GET'])
def download_report():
    data = session.get('last_analysis_data')

    if not data:
        flash("No recent analysis found to generate a report.", "warning")
        return redirect(url_for('home'))

    buffer = generate_pdf_report(data, title="MindSpace Social Analysis Report")

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{data['user']}_mindspace_report.pdf"
    )



# --- NEW PDF Download Route for HISTORY ---
@app.route('/download_history_report/<int:history_id>', methods=['GET'])
@login_required
def download_history_report(history_id):
    history_record = AnalysisHistory.query.filter_by(
        id=history_id,
        user_id=current_user.id
    ).first()

    if not history_record:
        flash("Report not found or access denied.", "danger")
        return redirect(url_for('history'))

    try:
        data = json.loads(history_record.full_results_json)
    except Exception:
        flash("Error loading report data from history.", "danger")
        return redirect(url_for('history'))

    buffer = generate_pdf_report(data, title="MindSpace Social Analysis Report (History)")

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{data['user']}_history_report_{history_id}.pdf"
    )



if __name__ == '__main__':
    app.run(debug=True)
