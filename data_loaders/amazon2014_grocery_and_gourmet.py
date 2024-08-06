from .amazon import Amazon


class Amazon2014GroceryAndGourmet(Amazon):

    @staticmethod
    def load_from_file(source_path, user_column_name, item_column_name, rating_column_name, timestamp_column_name,
                       **additional_parameters):
        return super(Amazon2014GroceryAndGourmet,
                     Amazon2014GroceryAndGourmet).load_from_file(source_path, user_column_name, item_column_name,
                                                                 rating_column_name, timestamp_column_name)
