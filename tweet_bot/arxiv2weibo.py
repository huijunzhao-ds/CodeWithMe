import requests, feedparser, time, logging
from datetime import date, timedelta, datetime
from wordcloud import WordCloud, STOPWORDS
import numpy as np 
from stop_words import stop_words
from access_token import token

logging.basicConfig(level=logging.DEBUG, 
                    filename='arxiv2weibo.log', 
                    filemode="a+", 
                    format="%(asctime)-15s %(levelname)-8s  %(message)s")


def search(date):
    posted_urls = []
    i, results_per_page = 0, 1000
    while True:
        arXiv_url = 'http://export.arxiv.org/api/query?search_query='
        query='(cat:stat.ML+OR+cat:cs.CV+OR+cat:cs.LG)&start=%d&max_results=%d'%(i, 
              results_per_page) + '&sortBy=submittedDate&sortOrder=descending'
        data = requests.get(arXiv_url+query).text
        entry_elements = feedparser.parse(data)['entries']

        # WARNING: more than 100 new papers per day can lead to temporary IP ban
        all_paper_info = []
        for val in entry_elements:
            paper_info = {
                'url': val['id'],
                'title': ' '.join(val['title'].split('\n')),
                'abstract': ' '.join(val['summary'].split('\n')),
                'date': val['published'].split('T')[0],
                'authors': [author['name'] for author in val['authors']],
            }
            
            if paper_info['date'] != str(date):
                return all_paper_info

            if paper_info['url'] not in posted_urls:
                all_paper_info.append(paper_info)
                posted_urls.append(paper_info['url'])
        i += results_per_page
        return all_paper_info


def text2wordcloud(texts:list):
    # define a circle mask
    x, y = np.ogrid[:500, :500]
    mask = (x - 250) ** 2 + (y - 250) ** 2 > 240 ** 2
    mask = 255 * mask.astype(int)

    all_texts = '\n'.join(texts)
    WC_engine = WordCloud(mask=mask, stopwords=list(STOPWORDS)+stop_words, 
                          background_color='white')
    
    wordcloud = WC_engine.generate(all_texts)
    return wordcloud


def share_weibo(text:str, img):
    safe_domain = 'https://arxiv.org'
    url_share = 'https://api.weibo.com/2/statuses/share.json'
    
    payload = {
        'access_token': token,
        'status':text + ' ' + safe_domain
    }
    if img :
        res = requests.post(url_share, data = payload, files = {"pic":img})
    else :
        res = requests.post(url_share, data = payload)
    return res


def clean_text(text: str):
    return " ".join(text.replace("\n", " ").split())


def post():
    dates = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_today = date.today().weekday()

    if day_today in [1, 2, 3, 4]:
        papers = search(date.today()-timedelta(days=1))
    elif day_today == 0:
        papers = search(date.today()-timedelta(days=3))
    else:
        papers = []
        return None

    logging.info('%s, parsed %d papers from arXiv'%(dates[day_today], len(papers)))
    for i, paper in enumerate(papers):
        logging.info('\n'.join([
            '\nPaper #%d, %s:' %(i+1, paper['url']),
            'Title: %s' %paper['title'],
            'Authors: %s' %(', '.join(paper['authors'])),
            'Abstract: %s' %paper['abstract']
        ]))

    abstracts = [paper['abstract'] for paper in papers] if len(papers)>0 else []
    if len(abstracts)>0:
        wc = text2wordcloud(abstracts)
        wc_file = 'wordclouds/wc_%s.png'%date.today()
        wc.to_file(wc_file)
        text = '''We generate a word cloud from the abstracts of {} papers on Machine
                  Learning posted last weekday on arXiv.'''.format(len(papers))
        res = share_weibo(clean_text(text), open(wc_file,'rb'))
        logging.info('Generated wordcloud, saved as %s'%wc_file)
    else:
        res = None

    return res 


if __name__=='__main__':
    next_start = datetime.now() 
    while True:
        if datetime.now() >= next_start:
            post()
        next_start += timedelta(days=1)
        time.sleep(24 * 60 * 60)