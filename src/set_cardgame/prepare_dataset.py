# '''
# code for preparing dataset for the SET game
# '''

# ### imports
# import os
# import numpy as np
# import pandas as pd
# import logging
# import glob


# # setup logger
# logger = logging.getLogger(__name__)


# # featurs and values for this dataset
# FEATURES_META = [
#     {
#         'name': 'color', 
#         'feature_column': 'f_color',
#         'target_column': 't_color',
#         'values': ['red', 'green', 'purple']}, 
#     {
#         'name': 'shape', 
#         'feature_column': 'f_shape', 
#         'target_column': 't_shape',
#         'values': ['round', 'squiggle', 'square']},
#     {
#         'name': 'fill', 
#         'feature_column': 'f_fill', 
#         'target_column': 't_fill',
#         'values': ['solid', 'dotted', 'open']},
#     {
#         'name': 'number', 
#         'feature_column': 'f_number',
#         'target_column': 't_number',
#         'values': ['one', 'two', 'three']}
# ]


# ### helper functions
# def string_to_dict(s, keys=None, sep="_"):
#     '''
#     maps a string separated by a character to a dictionary. 
    
#     For example: 
#         >> keys = ['color', 'shape', 'fill', 'number']
#         >> string_to_dict("green_square_solid_one", keys=keys)
#         >> {'color': 'green', 'shape': 'square', 'number': 'one'}
#     '''
#     try:
#         values = s.split(sep)
        
#         assert len(values) == len(keys), f"nr keys ({len(keys)}) does not match nr values in filename ({len(values)}) for '{s}'"
        
#         return dict(zip(keys, values))
#     except Exception as e:
#         logger.warning(f"string_to_dict: {e} while processing: {s}")


# def filename_to_dict(filename, keys, check_valid_image=False):
#     '''
#     maps a filename 
    
#     For example: 
#         >> keys = ['variant', 'color', 'shape', 'fill', 'number']
#         >> filename_to_target("0_purple_square_dotted_three.jpg", keys=keys)
#         >> {'variant': '0', 'color': 'purple', shape': 'square', 'fill': 'dotted', 'number': 'three'}
#     '''
#     try:
#         image_name, ext = filename.split(".")
    
#         if check_valid_image:
#             if not ext.lower() in ["jpg", "jpeg", "png"]:
#                 logger.debug(f"filename_to_dict:: {filename} not a valid image")
#                 return None
#         return string_to_dict(image_name, keys)
#     except Exception as e:
#         logger.warning(f"filename_to_dict: {e} while processing: {filename}")


# def query_from_dict(d):
#     query = []
#     for k, v in d.items():
#         query.append(f"({k}=='{v}')")
#     return "&".join(query)


# def create_metadata(dataset_path, csv_path):
#     '''
#     saves CSV with meta data for the dataset, created by parsing filenames that contains the meta data

#     parameters:
#     :dataset_path pathlib.Path path to dataset, should contain just image files, with the pattern: "{variant}_{color}_{shape}_{fill}_{number}.jpg"
#     :csv_path pathlib.Path 
#     '''

#     # create keys for that matches image filenames
#     # For example: filename '2_purple_square_dotted_three.jpg' matches ['variant', 'color', 'shape', 'fill', 'number']
#     keys = ['variant']
#     keys.extend([f['feature_column'] for f in FEATURES_META])
    
#     # create columns for resulting dataframe
#     columns = ['filename', 'variant']
#     columns.extend([f['feature_column'] for f in FEATURES_META])
    

#     # parse all filenames in dataset and create dataframe
#     output = []

#     for path in glob.glob(str(dataset_path / '*/*/*')):
#         filename = os.path.basename(path)
#             filename_dict = filename_to_dict(filename, keys=keys, check_valid_image=True)
#         if filename_dict is None:
#             logger.warning(f"skipping file: {filename}, could not process filename correctly")
#             continue
#         filename_dict['filename'] = filename
#         output.append(filename_dict)
        
#     # create dataframe
#     df_dataset = pd.DataFrame(data=output, columns=columns)

#     logger.info(f"created dataframe with metadata")
#     logger.debug(f"nr of records: {len(df_dataset)}")
#     logger.debug(f"columns: {columns}")


#     # create target columns (feature encodings)
#     for f in FEATURES_META:
#         values = f['values']
        
#         # use index as the ordinal integer (ie. red: 0,green:1,purpl:2) and create target column for each feature
#         df_dataset[f['target_column']] = df_dataset[f['feature_column']].apply(lambda x: values.index(x))

#     # sort dataframe
#     df_dataset = df_dataset.sort_values(['t_color', 't_shape', 't_fill', 't_number', 'variant']).reset_index(drop=True)
        
#     # create column card_id, initialize to 0 (to make sure it will be an integer)
#     df_dataset['card_id'] = 0

#     # create unique card id. There are multiple variants per card, so, loop through the variants
#     for variant in df_dataset.variant.unique():
#         # filter on variant
#         cond = (df_dataset['variant'] == variant)
        
#         # assign unique id
#         df_dataset.loc[cond, 'card_id'] = np.arange(len(df_dataset[cond]))
    
#     # export dataset to csv
#     df_dataset.to_csv(csv_path, index=False)

#     logger.info(f"saved dataframe with metadata to {csv_path}")
