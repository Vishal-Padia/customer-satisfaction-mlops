import logging
import pandas as pd

from zenml import step


class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting data from the data_path
        """
        logging.info(f"Ingesting the data from {self.data_path}")
        return pd.read_csv(self.data_path)


@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data_path

    Args:
        data_path: path to the data
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting the data {e}")
        raise e
