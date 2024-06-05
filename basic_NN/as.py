import requests
from bs4 import BeautifulSoup

def get_vietnamese_text(url, num_paragraphs=100):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ''
    for i in range(min(num_paragraphs, len(paragraphs))):
        text += paragraphs[i].text + '\n'
    return text

url = 'https://vi.wikipedia.org/wiki/Trang_Ch%C3%ADnh'
vietnamese_text = get_vietnamese_text(url)
print(vietnamese_text)
