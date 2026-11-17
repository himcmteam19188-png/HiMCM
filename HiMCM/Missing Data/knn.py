from pathlib import Path
from typing import Mapping

import numpy as np
import polars as pl


PREDICTORS = ("cost_of_living", "crime_rate", "restaurant_index")
CRITERIA = (
    'mental_health',
)

def read_csv(path: str | Path) -> pl.DataFrame:
    frame = pl.read_csv(path, encoding="utf-8-sig")
    frame = frame.rename({column: column.strip() for column in frame.columns})
    frame = frame.with_columns(pl.col('city').cast(pl.String).str.strip_chars().map_elements(lambda value: value.casefold(), return_dtype=pl.String).alias('_city'))
    frame = frame.with_columns(pl.col('state').cast(pl.String).str.strip_chars().str.to_uppercase().alias('_state'))
    return frame


def numeric(series: pl.Series) -> pl.Series:
    cleaned = series.cast(pl.String).str.replace_all(r"^\s*$", "").replace('', None)
    return cleaned.cast(pl.Float64, strict=True)


def metric(path: str | Path, name: str) -> pl.DataFrame:
    frame = read_csv(path)
    return frame.select(['_city', '_state', numeric(frame['est']).alias(name)])


def impute(cities_csv: str | Path, data_files: Mapping[str, str | Path], output_csv: str | Path | None, *, k: int, preserve_known: bool) -> pl.DataFrame:
    cities = read_csv(cities_csv)
    result = cities.clone()
    keys = ['_city', '_state']

    query = cities.select(keys + [numeric(cities[name]).alias(name) for name in PREDICTORS])
    features = metric(data_files[PREDICTORS[0]], PREDICTORS[0])
    for name in PREDICTORS[1:]:
        features = features.join(metric(data_files[name], name), on=keys, how='inner')
    features = features.join(cities.select(keys), on=keys, how='anti')

    for criterion in CRITERIA:
        observed = metric(data_files[criterion], criterion)
        known = cities.select(keys).join(observed.select(keys + [criterion]), on=keys, how='left').get_column(criterion)
        needs_prediction = known.is_null() if preserve_known else pl.Series('needs_prediction', [True] * cities.height)
        estimates = known.clone() if preserve_known else pl.Series(criterion, [np.nan] * cities.height)
        training_values = observed.join(cities.select(keys), on=keys, how='anti').drop_nulls(criterion)

        if needs_prediction.any():
            training = features.join(training_values, on=keys, how="inner")
            x_train = training.select(PREDICTORS).to_numpy()
            x_query = query.filter(needs_prediction).select(PREDICTORS).to_numpy()
            # fit scaling only on eligible donors not on held-out queries.
            center = x_train.mean(axis=0)
            spread = x_train.std(axis=0, ddof=0)
            spread[spread == 0] = 1.0  # constant predictors contribute no ranking information.
            x_train = (x_train - center) / spread
            x_query = (x_query - center) / spread
            targets = training.get_column(criterion).to_numpy()
            predictions = []
            for row in x_query:
                # mean distance
                distances = np.abs(x_train - row).mean(axis=1)
                nearest = np.argsort(distances, kind="stable")[:k]
                predictions.append(targets[nearest].mean())
            values = np.array(estimates.to_numpy(), copy=True)
            values[np.flatnonzero(needs_prediction.to_numpy())] = predictions
            estimates = pl.Series(criterion, values)

        result = result.with_columns(estimates.alias(criterion), needs_prediction.alias(f'{criterion}_imputed'))

    result = result.drop(keys)
    if output_csv is not None:
        result.write_csv(output_csv)
    return result


def predict_knn(cities_csv: str | Path, data_files: Mapping[str, str | Path], output_csv: str | Path | None = None, *, k: int = 10, preserve_known: bool = True,) -> pl.DataFrame:
    return impute(cities_csv, data_files, output_csv, k=k, preserve_known=preserve_known)


def build_csv(file_path: str, locations: list[str]):
    populations = pl.read_csv('Data/Final Data/population_density.csv')
    cost_of_living = pl.read_csv('Data/Final Data/cost_of_living.csv')
    crime_rate = pl.read_csv('Data/Final Data/crime_rate.csv')
    restaurant_index = pl.read_csv('Data/Final Data/restaurant_index.csv')
    with open(file_path, 'w') as f:
        f.write('city,state,state_long,population,cost_of_living,crime_rate,restaurant_index\n')
        for location in locations:
            city, abbrev, state = location.split(',')
            population = populations.filter((pl.col('city') == city) & (pl.col('state') == abbrev))
            population = population[0, 4]
            cost = cost_of_living.filter((pl.col('city') == city) & (pl.col('state') == abbrev))
            cost = cost[0, 3]
            crime = crime_rate.filter((pl.col('city') == city) & (pl.col('state') == abbrev))
            crime = crime[0, 3]
            restaurant = restaurant_index.filter((pl.col('city') == city) & (pl.col('state') == abbrev))
            restaurant = restaurant[0, 3]
            f.write(f'{city},{abbrev},{state},{population},{cost},{crime},{restaurant}\n')



if __name__ == "__main__":
    CITIES_CSV = r"Missing Data/mental_health_missing.csv"
    DATA_FILES = {
        "cost_of_living": r"Data/Final Data/cost_of_living.csv",
        "crime_rate": r"Data/Final Data/crime_rate.csv",
        "restaurant_index": r"Data/Final Data/restaurant_index.csv",
        'mental_health': r'Data/mental_health.csv'
    }
    knn = predict_knn(CITIES_CSV, DATA_FILES, "knn_predictions.csv")
    print("Wrote knn_predictions.csv")