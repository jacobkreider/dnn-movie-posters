#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import os.path
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image

from keras.callbacks import *
from clr_callback import *
from keras.optimizers import Adam

os.chdir('/home/jacob/data/model/')


# In[2]:


images_folder = '/home/jacob/data/images/'
test_data_ratio = 7  # 14.3%
validation_data_ratio = 6  # 14.3%
parsed_movies = []  # cache


class Movie:
    imdb_id = 0
    title = ''
    year = 0
    genres = []
    poster_url = ''

    def poster_file_exists(self) -> bool:
        return os.path.isfile(self.poster_file_path())

    def download_poster(self):
        try:
            response = urllib.request.urlopen(self.poster_url)
            data = response.read()
            file = open(self.poster_file_path(), 'wb')
            file.write(bytearray(data))
            file.close()
            return data
        except:
            print('-> error')

    def poster_file_path(self, size=100) -> str:
        return images_folder + str(size) + "/" + self.poster_file_name()

    def poster_file_name(self):
        return str(self.imdb_id) + '.jpg'

    def is_valid(self) -> bool:
        return self.poster_url.startswith('https://')                and 1995 <= self.year <= 2019                and len(self.title) > 1                and len(self.genres) > 1

    def to_rgb_pixels(self, poster_size):
        data = open(images_folder + str(poster_size) + '/' + str(self.imdb_id) + '.jpg', "rb").read()
        image = Image.open(io.BytesIO(data))
        rgb_im = image.convert('RGB')
        pixels = []
        for x in range(image.size[0]):
            row = []
            for y in range(image.size[1]):
                r, g, b = rgb_im.getpixel((x, y))
                pixel = [r / 255, g / 255, b / 255]
                row.append(pixel)
            pixels.append(row)

        return pixels

    def get_genres_vector(self, genres):
        if len(genres) == 1:
            has_genre = self.has_genre(genres[0])
            return [int(has_genre), int(not has_genre)]
        else:
            vector = []
            if self.has_any_genre(genres):
                for genre in genres:
                    vector.append(int(self.has_genre(genre)))

            return vector

    def short_title(self) -> str:
        max_size = 20
        return (self.title[:max_size] + '..') if len(self.title) > max_size else self.title

    def is_test_data(self) -> bool:
        return self.imdb_id % test_data_ratio == 0

    def has_any_genre(self, genres) -> bool:
        return len(set(self.genres).intersection(genres)) > 0

    def has_genre(self, genre) -> bool:
        return genre in self.genres

    def __str__(self):
        return self.short_title() + ' (' + str(self.year) + ')'


def download_posters(min_year=0):
    for movie in list_movies():
        print(str(movie))
        if movie.year >= min_year:
            if not movie.poster_file_exists():
                movie.download_poster()
                if movie.poster_file_exists():
                    print('-> downloaded')
                else:
                    print('-> could not download')
            else:
                print('-> already downloaded')
        else:
            print('-> skip (too old)')


def load_genre_data(min_year, max_year, genres, ratio, data_type, verbose=True):
    xs = []
    ys = []

    for year in reversed(range(min_year, max_year + 1)):
        if verbose:
            print('loading movies', data_type, 'data for', year, '...')

        xs_year, ys_year = _load_genre_data_per_year(year, genres, ratio, data_type)
        _add_to(xs_year, xs)
        _add_to(ys_year, ys)

        if verbose:
            print('->', len(xs_year))

    return np.concatenate(xs), np.concatenate(ys)


def _load_genre_data_per_year(year, genres, poster_ratio, data_type):
    xs = []
    ys = []

    count = 1
    for movie in list_movies(year, genres):
        if movie.poster_file_exists():
            if (data_type == 'train' and not movie.is_test_data() and count % validation_data_ratio != 0)                     or (data_type == 'validation' and not movie.is_test_data() and count % validation_data_ratio == 0)                     or (data_type == 'test' and movie.is_test_data()):
                x = movie.to_rgb_pixels(poster_ratio)
                y = movie.get_genres_vector(genres)
                xs.append(x)
                ys.append(y)
            count += 1

    xs = np.array(xs, dtype='float32')
    ys = np.array(ys, dtype='uint8')
    return xs, ys


def _add_to(array1d, array2d):
    if len(array1d) > 0:
        array2d.append(array1d)


def list_movies(year=None, genres=None):
    if len(parsed_movies) == 0:
        data = pd.read_csv('/home/jacob/data/MovieGenre.csv', encoding='ISO-8859-1')
        for index, row in data.iterrows():
            movie = _parse_movie_row(row)
            if movie.is_valid():
                parsed_movies.append(movie)

        parsed_movies.sort(key=lambda m: m.imdb_id)

    result = parsed_movies

    if year is not None:
        result = [movie for movie in result if movie.year == year]

    if genres is not None:
        result = [movie for movie in result if movie.has_any_genre(genres)]

    return result


def _parse_movie_row(row) -> Movie:
    movie = Movie()
    movie.imdb_id = int(row['imdbId'])
    movie.title = row['Title'][:-7]
    year = row['Title'][-5:-1]
    if year.isdigit() and len(year) == 4:
        movie.year = int(row['Title'][-5:-1])

    url = str(row['Poster'])
    if len(url) > 0:
        movie.poster_url = url.replace('"', '')

    genre_str = str(row['Genre'])
    if len(genre_str) > 0:
        movie.genres = genre_str.split('|')

    return movie


def search_movie(imdb_id=None, title=None) -> Movie:
    movies = list_movies()
    for movie in movies:
        if imdb_id is not None and movie.imdb_id == imdb_id:
            return movie
        if title is not None and movie.title == title:
            return movie


def list_genres(number):
    if number == 3:
        return ['Comedy', 'Drama', 'Action']
    if number == 7:
        return list_genres(3) + ['Animation', 'Romance', 'Adventure', 'Horror']
    if number == 14:
        return list_genres(7) + ['Sci-Fi', 'Crime', 'Mystery', 'Thriller', 'War', 'Family', 'Western']


# In[3]:


import os
import time

import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential


# In[ ]:



def get_kernel_dimensions(version, shape, divisor):
    image_width = shape[1]

    # original
    if version == 1:
        return 3, 3

    # square 10% width
    if version == 2:
        return int(0.1 * image_width / divisor), int(0.1 * image_width / divisor)

    # square 20% width
    if version == 3:
        return int(0.2 * image_width / divisor), int(0.2 * image_width / divisor)


def build(version, min_year, max_year, genres, ratio, epochs,
          x_train=None, y_train=None, x_validation=None, y_validation=None):
    # log
    print()
    print('version:', version)
    print('min_year:', min_year)
    print('max_year:', max_year)
    print('genres:', genres)
    print('ratio:', ratio)
    print()

    # load data if not provided
    if x_train is None or y_train is None or x_validation is None or y_validation is None:
        begin = time.time()
        x_train, y_train = load_genre_data(min_year, max_year, genres, ratio, 'train')
        x_validation, y_validation = load_genre_data(min_year, max_year, genres, ratio, 'validation')
        print('loaded in', (time.time() - begin) / 60, 'min.')
    else:
        print('data provided in arguments')

    print()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_validation.shape[0], 'validation samples')

    # build model
    num_classes = len(y_train[0])
    kernel_dimensions1 = get_kernel_dimensions(version, x_train.shape, 1)
    kernel_dimensions2 = get_kernel_dimensions(version, x_train.shape, 2)
    print('kernel_dimensions1:', kernel_dimensions1)
    print('kernel_dimensions2:', kernel_dimensions2)

    
    clr_triangular = CyclicLR(mode='triangular2', base_lr = .000025, max_lr = .0001)

    model = Sequential([
        Conv2D(32, kernel_dimensions1, padding='same', input_shape=x_train.shape[1:], activation='relu'),
        Conv2D(32, kernel_dimensions1, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, kernel_dimensions2, padding='same', activation='relu'),
        Conv2D(64, kernel_dimensions2, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')
    ])

    opt = keras.optimizers.rmsprop(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
    print(model.summary())

    model.fit(x_train, y_train, batch_size=32, epochs=epochs,callbacks=[clr_triangular]
                                , validation_data=(x_validation, y_validation))

    # create dir if none
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # save model
    model_file_name = 'Reduced epochs-genres'                       + '_' + str(min_year) + '_' + str(max_year)                       + '_g' + str(len(genres))                       + '_r' + str(ratio)                       + '_e' + str(epochs)                       + '_v' + str(version) + '.h5'

    model_path = os.path.join(save_dir, model_file_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


# In[ ]:


# Main

min_year = 1995
max_year = 2017
epochs = 10
genres = list_genres(7)

# select a smaller ratio (e.g. 40) for quicker training
for ratio in [30]:
    # we load the data once for each ratio, so we can use it for multiple versions, epochs, etc.
    x_train, y_train = load_genre_data(min_year, max_year, genres, ratio, 'train')
    x_validation, y_validation = load_genre_data(min_year, max_year, genres, ratio, 'validation')
    for version in [1]:
        build(version, min_year, max_year, genres, ratio, epochs,
                                 x_train=x_train,
                                 y_train=y_train,
                                 x_validation=x_validation,
                                 y_validation=y_validation)
