from .konect import Konect


class StackOverflow(Konect):

    @staticmethod
    def load_from_file(source_path, user_column_name, item_column_name, rating_column_name, timestamp_column_name,
                       **additional_parameters):
        version = "stackexchange-stackoverflow"
        has_timestamp = True
        return super(StackOverflow, StackOverflow).load_from_file(source_path, user_column_name, item_column_name,
                                                           rating_column_name, timestamp_column_name, version=version,
                                                           has_timestamp=has_timestamp)
