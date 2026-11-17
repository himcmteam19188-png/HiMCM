import requests
from bs4 import BeautifulSoup


def simple_niche_scrape(session: requests.Session, page: int):
    url = f'https://www.niche.com/colleges/search/top-public-universities/?page={page}'
    rankings = {}
    response = session.get(url)

    if (response.status_code == 403):
        print(f'Denied on page {page}')
        return {}
    elif (response.status_code == 200):
        soup = BeautifulSoup(response.text, 'html.parser')
        for college in soup.find_all('div', class_='MuiTypography-root MuiTypography-preheaderSmall search-result-badge nss-qu9x70'):
            # this is the ranking text of the college
            college_rank = college.text.lower().strip().split(' ')[0][1:] # first one is the number, ignore the hashtag
            college_card = college.parent
            if (college_card is None):
                print('no college card')
                continue
            # now we need to get the state (we don't care abt city for this)
            location = college_card.find('li', class_='MuiTypography-root MuiTypography-bodyXSmall search-result-tagline__item nss-s0wr6d')
            if (location is None):
                print('no location')
                continue
            try:
                state = location.text.lower().strip().split(', ')[1]
                rankings[int(college_rank)] = state
            except IndexError:
                    rankings[int(college_rank)] = location
    
        return rankings

    return {}
        

def scrape_athletics(session: requests.Session, page: int):
    url = f'https://highschoolrank.com/sports-rankings/'
    rankings = []
    response = session.get(url)

    if (response.status_code == 403):
        print(f'Denied on page {page}')
        return []
    elif (response.status_code == 200):
        soup = BeautifulSoup(response.text, 'html.parser')
        for ranking in soup.find_all('span', class_='inline-flex items-center justify-center rounded-full font-extrabold tabular-nums shadow-sm h-9 w-9 border border-slate-200 bg-slate-50 text-sm text-slate-500'):
            rank = int(ranking.text)
            ranking_row = ranking.parent
            if (ranking_row is None):
                print('no parent 1')
                continue
            ranking_row = ranking_row.parent
            if (ranking_row is None):
                print('no parent 2')
                continue
            location = ranking_row.find('span', class_='block truncate text-xs text-slate-500')
            if (location is None):
                print('no parent 3')
                continue
            location = location.text.split(' · ')[0].replace(', ', ',')
            rankings.append({'location': location, 'rank': rank})

        return rankings

    return {}


if __name__ =='__main__':
    session = requests.Session()
    session.headers.update({'User-Agent': 'SchoolResearch/1.0'})
    with open('Data/athletics.csv', 'a') as f:
        f.write('city,state,rank\n')
    with session:
        new_rankings = scrape_athletics(session, 1)
        with open('Data/athletics.csv', 'a') as f:
            for ranking in new_rankings:
                f.write(f'{ranking['location']},{ranking['rank']}\n')
