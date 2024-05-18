import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
from tqdm import tqdm


def fetch_documents(date, edinet_code):
    load_dotenv()
    subscription_key = os.getenv('EDINET_SUBSCRIPTION_KEY')

    url = f"https://api.edinet-fsa.go.jp/api/v2/documents.json?date={date}&Subscription-Key={subscription_key}&type=2"
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        print(
            f"Error: Unable to fetch data from API for date {date}. Status Code: {response.status_code}")
        return

    if data['metadata']['resultset']['count'] > 0:
        documents = data.get('results', [])
        if edinet_code:
            doc_ids = [(doc['docID'], doc['docDescription'], doc['edinetCode'])
                       for doc in documents if doc['edinetCode'] == edinet_code]
        else:
            doc_ids = [(doc['docID'], doc['docDescription'],
                        doc['edinetCode']) for doc in documents]
        download_pdfs(doc_ids, date)


def download_pdfs(doc_ids, date):
    base_url = "https://disclosure2dl.edinet-fsa.go.jp/searchdocument/pdf"

    for doc_id, doc_description, edinet_code in doc_ids:
        safe_description = doc_description.replace("/", "_").replace("\\", "_")
        save_path = f"documents/{edinet_code}/{date}"
        os.makedirs(save_path, exist_ok=True)

        pdf_url = f"{base_url}/{doc_id}.pdf"
        pdf_response = requests.get(pdf_url)

        if pdf_response.status_code == 200:
            pdf_file_path = os.path.join(save_path, f"{safe_description}.pdf")
            with open(pdf_file_path, 'wb') as pdf_file:
                pdf_file.write(pdf_response.content)
            print(f"Downloaded {doc_id}.pdf as {safe_description}.pdf")
        else:
            print(
                f"Error: Unable to download PDF for docID: {doc_id}. Status Code: {pdf_response.status_code}")


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


if __name__ == "__main__":
    start_date = datetime.strptime("2023-05-19", "%Y-%m-%d")
    # Adjusted to cover the last year
    end_date = datetime.strptime("2024-05-18", "%Y-%m-%d")
    edinet_code = input(
        "Enter the edinetCode (leave empty to download for all edinetCodes): ")

    total_days = (end_date - start_date).days + 1

    for single_date in tqdm(daterange(start_date, end_date), total=total_days, desc="Processing dates", unit="date"):
        date_str = single_date.strftime("%Y-%m-%d")
        fetch_documents(date_str, edinet_code)
