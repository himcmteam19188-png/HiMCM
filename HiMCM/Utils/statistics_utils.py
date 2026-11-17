from typing import Optional
from pathlib import Path

import polars as pl
import numpy as np


def _normalize_indicators(df: pl.DataFrame, indicators: list[int]) -> pl.DataFrame:
    """Normalize a dataframe according to the provided indicators list

    :param df: A polars dataframe containing the data
    :type df: pl.DataFrame
    :param indicators: A list containing whether each column should be positively normalized (a non-zero value, largest value maps to 1) or negatively normalized (zero, smallest value maps to 1)
    :type indicators: list[int]
    :return: The normalized dataframe
    :rtype: pl.DataFrame
    """
    maxes = df.max()

    for i, column in enumerate(df.columns):
        df = df.with_columns(pl.when(indicators[i] == 0).then(maxes[column] - pl.col(column)).otherwise((pl.col(column)) / pl.col(column).sum()))
        df = df.with_columns(pl.when(indicators[i] == 0).then(pl.col(column) / pl.col(column).sum()).otherwise(pl.col(column)))
    return df


def get_ewm_weights(positive_indicators: list[int], from_file: bool = True, file_path: Optional[str] = None, df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    # argument checking
    if (from_file and not isinstance(file_path, str)):
        raise ValueError('If from_file is True, then a file_path (of type str) must be passed')
    if (not from_file and not isinstance(df, pl.DataFrame)):
        raise ValueError('If from_file is False, then a df (of type polars.DataFrame) must be passed')
    if (from_file and file_path is not None):
        true_path = Path(file_path)
        if (true_path.suffix != '.csv'):
            raise ValueError('file_path must be a path to a .csv file')

    if (from_file and file_path is not None):
        df = pl.read_csv(file_path)

    if (df is None):
        raise ValueError('The data in file_path was not able to be convereted into a polars.DataFrame')
    if (len(positive_indicators) != df.width):
        raise ValueError('positive_indicators must have the same length as the width of the dataframe')

    # we begin by normalizing all of the data so it is between 0 and 1
    # each column is its own criteria
    # we need to know whether we want higher or lower values for each before we can normalize
    # if positive_indicators[i] is nonzero for column i, then it is a positive indicator
    df = _normalize_indicators(df, positive_indicators)

    entropy_constant = -1 / np.log(df.height) # h = -1/ln(m)
    entropy_values = {}
    # we need to replace all the 0s so logs don't break. Replace with 1 so logs evaluate to 0
    df = df.with_columns(pl.all().replace(0, 1))
    for column in df.columns:
        df = df.with_columns(pl.col(column) * pl.col(column).log())

    for column in df.columns:
        entropy_values[column] = entropy_constant * sum(df.get_column(column))

    entropy_values = pl.DataFrame(entropy_values)
    for column in entropy_values.columns:
        entropy_values = entropy_values.with_columns(1 - pl.col(column)) # convert to diversity degree
    entropy_values = entropy_values.drop('literal') # for some reason there's an extra column

    # finally convert to weights by finding proportions again
    total = entropy_values.sum_horizontal()
    entropy_values = entropy_values / total

    return entropy_values


def combine_data():
    files = sorted(Path('Data/Final Data').glob('*.csv'))
    combined = pl.read_csv(files[0]).select(['city', 'state', 'state_long', pl.col('est').alias(files[0].stem)])

    for file_path in files[1:]:
        if (file_path.stem == 'university_quality'):
            # this means we have to join separately since it is by state, not by city
            current = pl.read_csv(file_path).select(['state', pl.col('est').alias(file_path.stem)])
            current = current.with_columns(pl.col('state').str.to_uppercase())
            combined = combined.join(current, on='state', how='left')
            continue
        if (file_path.stem == 'weights'):
            continue

        current = pl.read_csv(file_path).select(['city', 'state', pl.col('est').alias(file_path.stem)])
        combined = combined.join(current, on=['city', 'state'], how='inner')

    combined.write_csv('Data/Final Data/combined.csv')


def print_top_cities_by_total(df: pl.DataFrame, output_path: str, print_res: bool = True, write: bool = False) -> pl.DataFrame:
    ignored_columns = {'city', 'state', 'state_long'}
    value_columns = [column for column in df.columns if column not in ignored_columns and df[column].dtype.is_numeric()]
    top_cities = df.with_columns(pl.sum_horizontal(*[pl.col(column) for column in value_columns],ignore_nulls=True,).alias('total')).sort('total', descending=True).select(['city', 'state', 'total'])
    if (write):
        top_cities.write_csv(output_path)
    top_cities = top_cities.head(10)

    if (print_res):
        print(top_cities)
    return top_cities


def top_cities(output_path: str, averages: list[float], print_res: bool = True, write: bool = False, weights_df: Optional[pl.DataFrame] = None):
    if (weights_df is None):
        weights_df = pl.read_csv('Data/Final Data/weights.csv').with_columns(pl.Series('average_weightings', averages))
        weights = weights_df.with_columns(pl.col(col).dot(pl.col('average_weightings')) for col in weights_df.columns if col != 'average_weightings').rows(named=True)[0]
    else:
        weights = weights_df
    city_data = pl.read_csv('Data/Final Data/combined.csv')
    city_data = city_data.with_columns(pl.col(col) * weights[col] for col in city_data.drop(['city', 'state', 'state_long']).columns)
    results = print_top_cities_by_total(city_data, output_path, print_res, write)

    return results


def get_subjective(mental_health: bool = False):
    if (not mental_health):
        SUBJECTIVE_OECD = {'civic_engagement': 0.07674559805707348, 'community': 0.09799635701275047, 'environment': 0.11123254401942928, 'jobs': 0.10673952641165757, 'housing': 0.1061323618700668, 'safety': 0.12374013357619916, 'education': 0.12058287795992716, 'health': 0.12774741955069824, 'work_life_balance': 0.12908318154219794}
    else:
        SUBJECTIVE_OECD = {'civic_engagement': (0.0632 / 0.9139), 'community': (0.0807 / 0.9139), 'environment': (0.0916 / 0.9139), 'jobs': (0.0879 / 0.9139), 'housing': (0.0874 / 0.9139), 'safety': (0.1019 / 0.9139), 'education': (0.0993 / 0.9139), 'life_satisfaction': (0.1101 / 0.9139), 'health': (0.1052 / 0.9139), 'work_life_balance': (0.1063 / 0.9139)}
    athletics = SUBJECTIVE_OECD['work_life_balance'] / 3
    city_budget = SUBJECTIVE_OECD['civic_engagement'] / 2
    col = SUBJECTIVE_OECD['housing']
    crime = SUBJECTIVE_OECD['safety']
    div = SUBJECTIVE_OECD['community'] / 2
    educ = SUBJECTIVE_OECD['education'] / 2
    environ = SUBJECTIVE_OECD['environment']
    job = SUBJECTIVE_OECD['jobs']
    health = SUBJECTIVE_OECD['health']
    restaurant = SUBJECTIVE_OECD['work_life_balance'] / 3
    social = SUBJECTIVE_OECD['work_life_balance'] / 3 + SUBJECTIVE_OECD['community'] / 2
    transport = SUBJECTIVE_OECD['civic_engagement'] / 2
    uni = SUBJECTIVE_OECD['education'] / 2
    if (not mental_health):
        weights = [athletics, city_budget, col, crime, div, educ, environ, job, health, restaurant, social, transport, uni]
    else:
        mental = SUBJECTIVE_OECD['life_satisfaction']
        weights = [athletics, city_budget, col, crime, div, educ, environ, job, mental, health, restaurant, social, transport, uni]
    with open('Data/Final Data/weights.csv', 'a') as f:
        for weight in weights:
            f.write(f'{weight},')


def sensitivity_analysis():
    # we want to test different average weightings
    # find max and min that keep palo alto #1
    # find max and min that keep top 10 the same
    max_one = 0
    min_one = 1
    max_all = 0
    min_all = 1
    max_same = 0
    min_same = 1
    default = top_cities('', [0.5, 0.5], print_res=False, write=False).select(['city', 'state'])
    default_cities = default['city'].to_list()
    default_states = default['state'].to_list()
    default_locations = [','.join([city, state]) for city, state in zip(default_cities, default_states)]
    default_locations = sorted(default_locations)
    for a in range(1001):
        test = top_cities('', [a / 1000, 1 - (a / 1000)], print_res=False, write=False).select(['city', 'state'])
        if (test.equals(default)):
            if (a / 1000 > max_all):
                max_all = a / 1000
            if (a / 1000 < min_all):
                min_all = a / 1000

        if (test[0].equals(default[0])):
            if (a / 1000 > max_one):
                max_one = a / 1000
            if (a / 1000 < min_one):
                min_one = a / 1000

        # we want to check if it has the same top 10, not necessarily same order
        test_cities = test['city'].to_list()
        test_states = test['state'].to_list()
        test_locations = [','.join([city, state]) for city, state in zip(test_cities, test_states)]
        test_locations = sorted(test_locations)
        if (test_locations == default_locations):
            if (a / 1000 > max_same):
                max_same = a / 1000
            if (a / 1000 < min_same):
                min_same = a / 1000         

    return min_one, max_one, min_all, max_all, min_same, max_same


def vary_oecd_weights(variation: float = 0.05) -> tuple[int, int]:
    # we want to disturb the subjective weights a little bit to model
    rng = np.random.default_rng()
    default = top_cities('', [0.5, 0.5], print_res=False, write=False).select(['city', 'state'])
    default_cities = default['city'].to_list()
    default_states = default['state'].to_list()
    default_locations = [','.join([city, state]) for city, state in zip(default_cities, default_states)]
    default_locations = sorted(default_locations)
    diffferent_order_count = 0
    different_cities_count = 0
    for _ in range(1000):
        weights = pl.read_csv('Data/Final Data/weights.csv')
        perturbations = rng.uniform(-variation, variation, size=weights.width)

        original = weights.row(1)
        varied = np.asarray(original, dtype=float) * (1 + perturbations)
        varied /= varied.sum()

        varied_row = weights.slice(1, 1).with_columns(pl.Series(column, [value]) for column, value in zip(weights.columns, varied))
        weights = weights.slice(0, 1).vstack(varied_row) # this is our new weights df that we can pass to the rating function
        weights = weights.with_columns(pl.all().mean()).with_row_index('idx').filter(~pl.col('idx').is_in([1])).drop('idx')
        trial = top_cities('', [0.5, 0.5], print_res=False, write=False, weights_df=weights).select(['city', 'state'])
        if (not trial.equals(default)):
            diffferent_order_count += 1

        trial_cities = trial['city'].to_list()
        trial_states = trial['state'].to_list()
        trial_locations = [','.join([city, state]) for city, state in zip(trial_cities, trial_states)]
        trial_locations = sorted(trial_locations)
        if (trial_locations != default_locations):
            different_cities_count += 1

    return diffferent_order_count, different_cities_count


if __name__ == '__main__':
    # combine_data()
    # df = pl.read_csv('Data/Final Data/combined.csv').drop('state', 'city', 'state_long') # add 0 for university quality of dc before running this
    # weights = get_ewm_weights([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], from_file=False, df=df)
    # weights.write_csv('Data/Final Data/weights.csv')
    # get_subjective(mental_health=False) # delete comma on last line after this
    # top_cities('top_cities_50_50_mental.csv', [0.5, 0.5], write=True)
    # min_o, max_o, min_a, max_a, min_s, max_s = sensitivity_analysis()
    # print(f'Range for same top 1: [{min_o}-{max_o}]')
    # print(f'Range for same top 10: [{min_a}-{max_a}]')
    # print(f'Range for same top 10 (not order): [{min_s}-{max_s}]')
    # wrong_order, wrong_cities = vary_oecd_weights(variation=0.05)
    # print(f'Proportion that were different order: {wrong_order / 1000}')
    # print(f'Proportion with different top 10 cities: {wrong_cities / 1000}')
    pass
