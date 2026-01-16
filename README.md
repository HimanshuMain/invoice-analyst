# 🧾 AI Invoice Analyst

A smart financial assistant that extracts data from invoices and allows you to chat with your spreadsheets using natural language. Built with Python, Streamlit, and Google Gemini models.

## 🚀 Features

* **Intelligent Extraction:** Upload any invoice (Image/PDF) and get a perfectly structured CSV. No hardcoded rules—it detects columns automatically.
* **Smart Analytics:** Upload a CSV dataset and ask questions like "Who is the top vendor?" or "Total spend in March?".
* **Auto-Correction:** Automatically cleans currency symbols and formats numbers for calculation.
* **Privacy Focused:** Processes data in memory without storing files permanently.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **AI Engine:** Google Gemini (2.5 Flash & Pro)
* **Data Processing:** Pandas
* **Language:** Python 3.10+

## 📦 How to Run

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your `.env` file with `GOOGLE_API_KEY`.
4. Run the app: `streamlit run app.py`
