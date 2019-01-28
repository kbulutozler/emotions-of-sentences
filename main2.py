from utilities import *


X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')

maxLen = len(max(X_train, key=len).split())

Y_oh_train = convert_to_one_hot(Y_train, C = 2)
Y_oh_test = convert_to_one_hot(Y_test, C = 2)

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)

model = build_model((maxLen,), word_to_vec_map, word_to_index)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)

model.fit(X_train_indices, Y_oh_train, epochs = 50, batch_size = 32, shuffle=True)

loss, acc = model.evaluate(X_test_indices, Y_oh_test)
print()
print("Test accuracy = ", acc)

save_model(model, 'my_model')