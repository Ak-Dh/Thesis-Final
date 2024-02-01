from jaad_data import JAAD

# Set the path to the main project directory
jaad_path = '.'

# Initialize JAAD object
imdb = JAAD(data_path=jaad_path)

# Extract and save images
# imdb.extract_and_save_images()

# imdb.generate_database()

# Update the data_opts dictionary with your desired values
data_opts = {'fstride': 1,
             'sample_type': 'all',
             'subset': 'high_visibility',
             'data_split_type': 'default',
             'seq_type': 'trajectory',
             'height_rng': [0, float('inf')],
             'squarify_ratio': 0,
             'min_track_size': 0,
             'random_params': {'ratios': None,
                               'val_data': True,
                               'regen_data': True},
             'kfold_params': {'num_folds': 5, 'fold': 1}}

# sequence_data = imdb.generate_data_trajectory_sequence('train',**data_opts)
detection_data = imdb.get_detection_data('train', 'yolo3', **data_opts)

