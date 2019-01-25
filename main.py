from utilities import *


X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')


Y_oh_train = convert_to_one_hot(Y_train, C = 2)
Y_oh_test = convert_to_one_hot(Y_test, C = 2)

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.300d.txt')

pred, W, b = model(X_train, Y_train, word_to_vec_map)

print("Training set:")
pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
print('Test set:')
pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)