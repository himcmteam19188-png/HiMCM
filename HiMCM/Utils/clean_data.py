import json
import re

import polars as pl


def json_to_csv(file_path: str, header: str) -> None:
    with open(file_path, 'r') as f:
        data = json.load(f)

    with open(file_path, 'w') as f:
        f.write(header + '\n')
        for key, value in data.items():
            f.write(f'{key},{value}\n')


def remove_commas_in_numbers(file_path: str) -> None:
    df = pl.read_csv(file_path)
    for column in df.columns:
        col = df.get_column(column)
        is_numeric = True
        for val in col:
            if bool(re.search(r'[^0-9,]', str(val))): # true if not a number
                is_numeric = False

        if (is_numeric):
            df = df.with_columns(pl.col(column).str.replace_all(',', '').cast(pl.Float64))

    df.write_csv(file_path)


def replace_populations() -> None:
    full_dataset = pl.read_csv('Data/bad_populations.csv')
    good_populations = pl.read_csv('Data/good_populations.csv')
    # we need to replace the population values
    # the issue is that the names don't match exactly
    for city_bad in full_dataset['City']:
        broken = False
        for city_good in good_populations['City']:
            cleaned_name = ", ".join(city_good.split(' city, ')) # remove city if it's there
            cleaned_name = ", ".join(cleaned_name.split(' town, ')) # remove town if it's there
            cleaned_name = ", ".join(cleaned_name.split(' village, ')) # remove village if it's there
            if (cleaned_name.lower().strip() == city_bad.lower().strip()):
                # now we need to replace the population value
                row_index_bad = full_dataset['City'].index_of(city_bad)
                row_index_good = good_populations['City'].index_of(city_good)
                if (row_index_good is None):
                    continue
                if (row_index_bad is None):
                    print(f'No match for city {city_bad} ({city_good})')
                    break
                old_pop = full_dataset[row_index_bad, 'Population']
                new_pop = good_populations[row_index_good, 'Population']
                full_dataset = full_dataset.with_columns(pl.col('Population').replace(old_pop, new_pop))
                broken = True
                break

        if (not broken):
            print(f'No match for city {city_bad}')

    # now we recalc density
    full_dataset = full_dataset.with_columns((pl.col('Population') / pl.col('Size')).alias('Density'))

    full_dataset.write_csv('Data/final_populations.csv')


def convert_uni_data_to_ranking():
    df = pl.read_csv('Data/uni_data.csv')
    # there are two columns, ranking, and state
    ranking_totals = {}
    states = [
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
            "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
    ]
    for state in states:
        ranking_totals[state.lower()] = [] # initialize values

    for row in df.iter_rows():
        ranking_totals[row[1]].append(row[0])

    for state in states:
        if state.lower() not in ranking_totals.keys():
            ranking_totals[state.lower()] = 0

    # now do 1/sum
    with open('Data/state_avg_uni_ranking.csv', 'w') as f:
        f.write('state,ranking_sum,reciprocal_sum')
        for state, ranking in ranking_totals.items():
            ranking.sort()
            # we only take average of top 3 so that states with more don't just have a higher average rating
            # no state has no universities in top 348 so no divide by 0 concerns
            avg_ranking = sum(ranking[:min(len(ranking), 3)]) / min(len(ranking), 3)
            if (avg_ranking == 0):
                reciprocal = 0
            else:
                reciprocal = 1 / avg_ranking
            f.write(f'{state},{avg_ranking},{reciprocal}\n')


def clean_city_names():
    df = pl.read_csv('Data/cities.csv')
    cities = []
    states = []
    for city_name in df['city']:
        cleaned_name = ", ".join(city_name.split(' city, ')) # remove city if it's there
        cleaned_name = ", ".join(cleaned_name.split(' town, ')) # remove town if it's there
        cleaned_name = ", ".join(cleaned_name.split(' village, ')) # remove village if it's there
        cleaned_name = ", ".join(cleaned_name.split(' municipality, ')) # remove municipality if it's there
        city, state = cleaned_name.split(', ')
        cities.append(city)
        states.append(state)

    df = df.with_columns(pl.Series('city', cities), pl.Series('state', states))

    df.write_csv('cities_new.csv')


def clean_events_data():
    # we want to remove any data we don't need and make sure we have data for every city
    event_data = pl.read_csv('Data/events.csv')
    city_list = pl.read_csv('Data/cities.csv')
    cleaned_cities = []
    cleaned_states = []
    cleaned_scores = []
    for city, state in city_list.iter_rows():
        # search for the city in the aqi_data
        city_index = event_data['city'].index_of(city)
        if (city_index is None):
            print(f'No event data for {city}')
            continue
        # this means we have data
        cleaned_cities.append(city)
        cleaned_states.append(state)
        cleaned_scores.append(event_data[city_index, 'events'])

    cleaned_data = pl.DataFrame({
        'city': cleaned_cities,
        'state': cleaned_states,
        'score': cleaned_scores,
    })
    cleaned_data.write_csv('Data/cleaned_events.csv')


def clean_crime_data():
    # this is a weird format
    city_list = pl.read_csv('Data/cities.csv')
    crime_data = pl.read_excel('Data/crime_data.xlsx')
    # we want column (D + I) * (100000 / C) = (__UNNAMED__3 + __UNNAMED__8) * (100000 / __UNNAMED__2)
    cleaned_cities = []
    cleaned_states = []
    cleaned_rates = []
    for city, state, _ in city_list.iter_rows():
        # we compare state with 'Table 8' and City with __UNNAMED__1
        # weird superscript thing with states
        state_indices = crime_data.with_row_index('idx').filter(pl.col('Table 8').is_in([state, f'{state}2'])).get_column('idx').to_list()
        found = False
        for idx in state_indices:
            crime_city = crime_data[idx, '__UNNAMED__1']
            if (crime_city == city):
                # do our calculations (crime rate per 100,000)
                # they put commas in their numbers cuz they're annoying
                violent_crime = float(crime_data[idx, '__UNNAMED__3'].replace(',', ''))
                property_crime = float(crime_data[idx, '__UNNAMED__8'].replace(',', ''))
                population = float(crime_data[idx, '__UNNAMED__2'].replace(',', ''))
                crime_rate = (violent_crime + property_crime) * (100000 / population)
                cleaned_cities.append(city)
                cleaned_states.append(state)
                cleaned_rates.append(crime_rate)
                found = True
                break

        if (not found):
            print(f'No data for {city}')

    cleaned_data = pl.DataFrame({
        'city': cleaned_cities,
        'state': cleaned_states,
        'score': cleaned_rates,
    })
    cleaned_data.write_csv('Data/cleaned_crime.csv')


def split_data(labels: list[str], file_path: str):
    df = pl.read_csv(file_path)
    needed_columns = ['state_abbr', 'geo_name', 'est', 'data_period']
    for label in labels:
        separate_data = df.select(*needed_columns, 'metric_name', 'group_name')
        # get rid of everything where metric_name isn't the label
        separate_data = separate_data.filter(pl.col('metric_name') == label)
        # now get rid of all groups that aren't "Total"
        separate_data = separate_data.filter(pl.col('group_name') == 'Total')
        separate_data = separate_data.select(*needed_columns)
        separate_data.write_csv(f'Data/city_health_dashboard/{label.replace(' ', '_').replace('/', '-')}.csv')


def clean_city_health_data(file_path: str):
    # we want to remove any data we don't need and make sure we have data for every city
    hs_data = pl.read_csv(f'Data/city_health_dashboard/{file_path}.csv')
    city_list = pl.read_csv('Data/cities.csv')
    cleaned_cities = []
    cleaned_states = []
    cleaned_scores = []
    for city, state in city_list.iter_rows():
        # search for the city in the aqi_data
        city_index = hs_data['geo_name'].index_of(city)
        if (city_index is None):
            print(f'No event data for {city}')
            continue
        # this means we have data
        cleaned_cities.append(city)
        cleaned_states.append(state)
        cleaned_scores.append(hs_data[city_index, 'est'])

    cleaned_data = pl.DataFrame({
        'city': cleaned_cities,
        'state': cleaned_states,
        'score': cleaned_scores,
    })
    cleaned_data.write_csv(f'Data/cleaned_{file_path}.csv')


def add_monthly_data(file_path: str):
    df = pl.read_csv(file_path)
    result = df.group_by(['state_abbr', 'geo_name']).agg(pl.col('est').sum())
    result.write_csv(file_path)


def normalize_city_health_data(file_path: str):
    df = pl.read_csv(file_path)
    max_val = df['est'].max()
    min_val = df['est'].min()
    df = df.with_columns((pl.col('est') - min_val) / (max_val - min_val)) # type: ignore it will always be a number
    df.write_csv(file_path)


def get_aqi_sum():
    location_keys = ['state_abbr', 'geo_name']
    co2 = pl.read_csv('Data/city_health_dashboard/co2.csv').select(*location_keys, pl.col('est').alias('co2'))
    ozone = pl.read_csv('Data/city_health_dashboard/ozone.csv').select(*location_keys, pl.col('est').alias('ozone'))
    particulate_matter = pl.read_csv('Data/city_health_dashboard/particulate_matter.csv').select(*location_keys, pl.col('est').alias('particulate_matter'))

    aqi = co2.join(ozone, on=location_keys, how='inner').join(particulate_matter, on=location_keys, how='inner').with_columns(((pl.col('co2') + pl.col('ozone') + pl.col('particulate_matter')) / 3).alias('est')).select(*location_keys, 'est')
    aqi.write_csv('Data/cleaned_aqi.csv')


def clean_aqi_data():
    df = pl.read_csv('Data/cleaned_aqi.csv')
    cities = pl.read_csv('Data/cities.csv')
    cleaned_cities = []
    cleaned_states = []
    cleaned_aqi = []
    for city, state, _ in cities.iter_rows():
        city_index = df['geo_name'].index_of(city)
        if (city_index is None):
            print(f'No aqi data for {city}')
            continue
        cleaned_cities.append(city)
        cleaned_states.append(state)
        cleaned_aqi.append(df[city_index, 'est'])

    cleaned_data = pl.DataFrame({
        'city': cleaned_cities,
        'state': cleaned_states,
        'est': cleaned_aqi
    })
    cleaned_data.write_csv('Data/cleaned_aqi.csv')


def get_diversity_sum():
    location_keys = ['state_abbr', 'geo_name']
    diversity = pl.read_csv('Data/city_health_dashboard/diversity.csv').select(*location_keys, pl.col('est').alias('diversity'))
    segregation = pl.read_csv('Data/city_health_dashboard/segregation.csv').select(*location_keys, pl.col('est').alias('segregation'))
    segregation = segregation.with_columns((1 - pl.col('segregation')).alias('segregation'))

    aqi = diversity.join(segregation, on=location_keys, how='inner').with_columns(((pl.col('diversity') + pl.col('segregation')) / 2).alias('est')).select(*location_keys, 'est')
    aqi.write_csv('Data/cleaned_diversity.csv')


def clean_diversity_data():
    df = pl.read_csv('Data/cleaned_diversity.csv')
    cities = pl.read_csv('Data/cities.csv')
    cleaned_cities = []
    cleaned_states = []
    cleaned_diversity = []
    for city, state, _ in cities.iter_rows():
        # search for the city in the aqi_data
        city_index = df['geo_name'].index_of(city)
        if (city_index is None):
            print(f'No budget data for {city}')
            continue
        # this means we have data
        cleaned_cities.append(city)
        cleaned_states.append(state)
        cleaned_diversity.append(df[city_index, 'est'])

    cleaned_data = pl.DataFrame({
        'city': cleaned_cities,
        'state': cleaned_states,
        'score': cleaned_diversity,
    })
    cleaned_data.write_csv('Data/cleaned_diversity.csv')  


def get_city_budgets():
    # we get to parse weird government id system (called COG i think)
    # of course it's also a text file so we can't even use polars so we get to parse 511000 lines in pure python
    # 1-2 = state (01 is alabama, in alphabetical except we randomly skip some number for some god forsaken reason) (skip 3, 7, 14, 43, 52, and 11 is DC)
    # 3 = type (we want 2 for city)
    # 4-6 = county (i don't think we care)
    # 7-12 = unit identifier (this is city id, we have to look up in a different text file)
    # 13-15 = item code (we have to add a bunch of them)
    # 16-27 = budget (in thousands) (leading 0s are replaced with spaces)
    # 28-31 = year (always 2024)
    # 32-33 = imputation type (i don't think we care, also normally 1 character)
    state_id_dict = {'01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC',
                     '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME',
                     '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
                     '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', '45': 'SC',
                     '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI', '56': 'WY'}
    city_ids = {}
    with open('Data/pid_2024.txt', 'r') as f:
        data = f.readlines()
        for line in data:
            if (line[2] != '2'):
                continue # skip non cities
            id = line[:12]
            city_name = line[12:75].strip()
            # we also want population so we can do per capita (117-125)
            population = int(line[116:125].strip())
            city_ids[id] = [city_name, population]

    city_budgets = {} # city name to budget
    with open('Data/financials_census.txt', 'r') as f:
        data = f.readlines()
        for i, line in enumerate(data):
            if (line[2] != '2'):
                continue # don't care if it's not a city
            fips_id = line[:12]
            city, population = city_ids[fips_id]
            state = state_id_dict[line[:2]]
            combined = f'{city},{state}'
            # we want to add everything from E, F, I, J, L, S plus Y53
            if (line[12] in ['E', 'F', 'I', 'J', 'L', 'S', 'Y']):
                if (line[12] == 'Y' and line[12:14] != 'Y53'):
                    continue # skip because not the right Y
                # get the dollar amount
                budget = int(line[15:27].strip()) * 1000 # remove the leading spaces
                # do per capita
                try:
                    city_budgets[combined] += budget / population
                except KeyError:
                    city_budgets[combined] = budget / population

    # now we can write the data to a csv file
    with open('Data/city_budgets.csv', 'w') as f:
        f.write('city,state,budget\n')
        for location, budget in city_budgets.items():
            # we need to clean up the location rq
            location = location.lower()
            location = location.replace(' city,', ',').replace(' town,', ',').replace(' village,', ',').replace(' municipality,', ',')
            f.write(f'{location},{budget}\n')


def clean_city_budgets():
    df = pl.read_csv('Data/city_budgets.csv')
    cities = pl.read_csv('Data/cities.csv')
    cleaned_cities = []
    cleaned_states = []
    cleaned_budgets = []
    for city, state, _ in cities.iter_rows():
        # search for the city in the aqi_data
        city_index = df['city'].index_of(city.lower().replace('.', '')) # census doesn't have the . for some reason
        if (city_index is None):
            print(f'No budget data for {city}')
            continue
        # this means we have data
        cleaned_cities.append(city)
        cleaned_states.append(state)
        cleaned_budgets.append(df[city_index, 'budget'])

    cleaned_data = pl.DataFrame({
        'city': cleaned_cities,
        'state': cleaned_states,
        'score': cleaned_budgets,
    })
    cleaned_data.write_csv('Data/city_budgets.csv')


def clean_athletic_data():
    df = pl.read_csv('Data/athletics.csv')
    cities = pl.read_csv('Data/cities.csv')

    average_ranks = df.group_by(['city', 'abbreviation']).agg(pl.col('rank').mean().alias('est'))

    # print cities that have no matching athletics data.
    missing = cities.join(average_ranks, on=['city', 'abbreviation'], how='anti')
    for city, state, _ in missing.iter_rows():
        print(f'No athletics data for {city}, {state}')

    cleaned_data = cities.join(average_ranks, on=['city', 'abbreviation'], how='inner', maintain_order='left')
    cleaned_data = cleaned_data.with_columns(pl.col('est').max() - pl.col('est'))

    cleaned_data.write_csv('Data/cleaned_athletics.csv')


def clean_restaurant_data(file_path: str):
    df = pl.read_csv(file_path)
    city_counts = {}
    for row in df.iter_rows():
        city_name = row[1]
        state = row[2]
        location = f'{city_name},{state}'
        try:
            city_counts[location] += 1
        except KeyError:
            city_counts[location] = 1

    states = []
    cities = []
    for key in city_counts.keys():
        city, state = key.split(',')
        cities.append(city)
        states.append(state)

    states = pl.Series('state', states)
    cities = pl.Series('city', cities)
    counts = pl.Series('count', city_counts.values())
    df = pl.DataFrame([cities, states, counts])
    df.write_csv(file_path)


def clean_restaurant_data_v2(file_path: str):
    cities = pl.read_csv('Data/cities.csv')
    df = pl.read_csv(f'Data/{file_path}')
    cleaned_cities = []
    cleaned_states = []
    cleaned_restaurant_counts = []
    for city, state, _ in cities.iter_rows():
        city_index = df['city'].index_of(city)
        if (city_index is None):
            print(f'No restaurant data for {city}')
            continue
        cleaned_cities.append(city)
        cleaned_states.append(state)
        cleaned_restaurant_counts.append(df[city_index, 'count'])

    cleaned_data = pl.DataFrame({
        'city': cleaned_cities,
        'state': cleaned_states,
        'count': cleaned_restaurant_counts,
    })
    cleaned_data.write_csv(f'Data/cleaned_{file_path}')


def per_capita_restaurants(file_path: str):
    location_keys = ['state', 'city']
    df = pl.read_csv(f'Data/{file_path}')
    city_pops = pl.read_csv('Data/Final Data/population_density.csv').select(*location_keys, pl.col('population'))
    df = df.join(city_pops, on=location_keys, how='inner')
    df = df.with_columns((pl.col('count') / pl.col('population')).alias('est'))
    df = df.select(['city', 'state', 'est'])
    df.write_csv(f'Data/cleaned_{file_path}')


def combine_restaurant_data():
    location_keys = ['city', 'state']
    mcdonalds = pl.read_csv('Data/cleaned_mcdonalds.csv').select(*location_keys, pl.col('est').alias('mcdonalds')).group_by(location_keys).agg(pl.col('mcdonalds').mean())
    subway = pl.read_csv('Data/cleaned_subway.csv').select(*location_keys, pl.col('est').alias('subway')).group_by(location_keys).agg(pl.col('subway').mean())
    starbucks = pl.read_csv('Data/cleaned_starbucks.csv').select(*location_keys, pl.col('est').alias('starbucks')).group_by(location_keys).agg(pl.col('starbucks').mean())
    crime = pl.read_csv('Data/Final Data/crime_rate.csv').select(*location_keys, 'state_long').unique(subset=location_keys)

    result = crime.join(mcdonalds, on=location_keys, how='left').join(subway, on=location_keys, how='left').join(starbucks, on=location_keys, how='left').with_columns([pl.col('mcdonalds').fill_null(0), pl.col('starbucks').fill_null(0), pl.col('subway').fill_null(0)]).with_columns((pl.col('mcdonalds') + pl.col('subway') + pl.col('starbucks')).alias('est')).select(['city', 'state_long', 'state', 'est'])
    result.write_csv('Data/final_restaurants.csv')


def count_hospitals():
    df = pl.read_csv('Data/ushospitals.csv')
    hospital_counts = {}
    for row in df.iter_rows(named=True):
        location = f'{row['city']},{row['state']}'
        try:
            hospital_counts[location] += 1
        except KeyError:
            hospital_counts[location] = 1

    states = []
    cities = []
    for key in hospital_counts.keys():
        city, state = key.split(',')
        cities.append(city)
        states.append(state)

    states = pl.Series('state', states)
    cities = pl.Series('city', cities)
    counts = pl.Series('count', hospital_counts.values())

    df = pl.DataFrame([cities, states, counts])
    df.write_csv('Data/hospitals.csv')


def per_capita_hospitals():
    # we need to make hospital count per capita
    location_keys = ['state', 'city']
    df = pl.read_csv('Data/hospitals.csv')
    city_pops = pl.read_csv('Data/Final Data/population_density.csv').select(*location_keys, pl.col('population'))
    city_pops = city_pops.join(df, on=location_keys, how='left').with_columns(pl.col('count').fill_null(0))
    cleaned = city_pops.with_columns((pl.col('count') / pl.col('population')).alias('hospitals'))
    cleaned.write_csv('Data/hospitals.csv')


def clean_populations():
    df = pl.read_csv('Data/final_populations.csv')
    # we need to split the city and state
    split_city = pl.col('City').str.split_exact(', ', 1)
    df = df.with_columns(split_city.struct.field('field_0').alias('city'), split_city.struct.field('field_1').alias('state')).drop('City')
    df = df.select(['city', 'state', 'density', 'population', 'size'])
    df.write_csv('Data/final_populations.csv')


def combine_public_health():
    location_keys = ['city', 'state']
    hospital_count = pl.read_csv('Data/hospitals.csv').select(*location_keys, pl.col('est').alias('hospitals'))
    checkups = pl.read_csv('Data/city_health_dashboard/checkup.csv').select(*location_keys, pl.col('est').alias('checkup'))
    df = hospital_count.join(checkups, on=location_keys, how='inner').with_columns(((pl.col('hospitals') + pl.col('checkup')) / 2).alias('est'))
    df = df.select(location_keys + ['est'])
    df.write_csv('Data/cleaned_health.csv')


def clean_public_health():
    cities = pl.read_csv('Data/cities.csv')
    df = pl.read_csv('Data/cleaned_health.csv')
    cleaned_cities = []
    cleaned_states = []
    cleaned_health = []
    for city, state, _ in cities.iter_rows():
        city_index = df['city'].index_of(city)
        if (city_index is None):
            print(f'No hospital data for {city}')
            continue
        cleaned_cities.append(city)
        cleaned_states.append(state)
        cleaned_health.append(df[city_index, 'est'])

    cleaned_data = pl.DataFrame({
        'city': cleaned_cities,
        'state': cleaned_states,
        'count': cleaned_health,
    })
    cleaned_data.write_csv('Data/cleaned_health.csv')


def per_capita_transport():
    location_keys = ['state', 'city']
    df = pl.read_csv('Data/cleaned_transportation_data.csv')
    city_pops = pl.read_csv('Data/Final Data/population_density.csv').select(*location_keys, pl.col('population'))
    df = df.join(city_pops, on=location_keys, how='inner')
    df = df.with_columns((pl.col('trips') / pl.col('population')).alias('est'))
    df = df.select(['city', 'state', 'est'])
    df.write_csv('Data/cleaned_transportation.csv')


def combine_transport():
    location_keys = ['city', 'state']
    trips = pl.read_csv('Data/cleaned_transportation.csv').select(*location_keys, pl.col('est').alias('trips'))
    walkability = pl.read_csv('Data/city_health_dashboard/walkability.csv').select(*location_keys, pl.col('est').alias('walkability'))
    df = trips.join(walkability, on=location_keys, how='inner').with_columns(((pl.col('trips') + pl.col('walkability')) / 2).alias('est'))
    df = df.select(location_keys + ['est'])
    df.write_csv('Data/cleaned_transport_v2.csv')


def clean_house_price():
    cities = pl.read_csv('Data/cities.csv')
    df = pl.read_csv('Data/zillow.csv')
    cleaned_cities = []
    cleaned_states = []
    cleaned_price = []
    for city, state, _ in cities.iter_rows():
        city_index = df['city'].index_of(city)
        if (city_index is None):
            print(f'No house price data for {city}')
            continue
        cleaned_cities.append(city)
        cleaned_states.append(state)
        cleaned_price.append(df[city_index, 'price'])

    cleaned_data = pl.DataFrame({
        'city': cleaned_cities,
        'state': cleaned_states,
        'est': cleaned_price,
    })
    cleaned_data.write_csv('Data/cleaned_zillow.csv')


def find_crime_missing_populations(file_path: str):
    populations = pl.read_csv(file_path).select(['city', 'state'])
    crime = pl.read_csv('Data/Final Data/crime_rate.csv').select(['city', 'state', 'state_long']).unique()
    missing = crime.join(populations, on=['city', 'state'], how='anti').select(['city', 'state', 'state_long'])

    for city, state, state_long in missing.iter_rows():
        print(f'{city},{state},{state_long}')


def pare_down(file_path: str):
    location_keys = ['state', 'city']
    crime = pl.read_csv('Data/Final Data/crime_rate.csv').select(*location_keys, 'state_long').unique(subset=location_keys)
    df = pl.read_csv(f'{file_path}.csv')
    cleaned = crime.join(df, on=location_keys, how='left').select(*location_keys, 'state_long', 'est').with_columns(pl.col('est').fill_null(0))
    cleaned.write_csv(f'{file_path}_v2.csv')


def combine_knn(file_path: str, column: str):
    knn_data = pl.read_csv('knn_predictions.csv')
    df = pl.read_csv(f'{file_path}.csv')
    rows_to_add = []
    for row in knn_data.iter_rows(named=True):
        rows_to_add.append({'est': row[column], 'state': row['state'], 'city': row['city'], 'state_long': row['state_long']})
    df = pl.concat([df, pl.DataFrame(rows_to_add)])
    df.write_csv(f'{file_path}_v2.csv')


def get_california_budgets():
    df = pl.read_csv('Data/california_budgets.csv')
    df = df.filter(pl.col('Fiscal Year') == 2024).select(pl.col('Entity Name'), pl.col('Expenditures Per Capita'))
    df.write_csv('Data/california_budgets.csv')
