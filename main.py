# -----------------------------------------------------------
# Оконное приложение для ОС Windows, демонстрирующее работу алгоритмов
# классификации текстов новостных статей
# -----------------------------------------------------------

# Импорт необходимых библиотек
from tkinter import *
import numpy as np
import tkinter.filedialog as fd
import tkinter.messagebox as mb
import regex as re
import razdel as rd
import pymorphy2
import string
import pickle
import tkinter.ttk as ttk
from nltk.corpus import stopwords
import csv
from sqlalchemy import create_engine, MetaData, Table, Column, String, select
import hashlib
import pyperclip as pc
import os
import tensorflow as tf


# Dummy-функция для TF-IDF векторизации
def dummy_fun(x):
    return x


# Класс NeuralNetwork, отвечающий за работу искусственной нейронной сети
class NeuralNetwork:
    # Инициализация ANN с помощью воспроизведения архитектуры модели и загрузки обученных весов
    def __init__(self):
        self.__shape = 1052394
        self.__inputLayer = tf.keras.Input(shape=self.__shape)
        self.__x = tf.keras.layers.Dense(1024, activation='relu')(self.__inputLayer)
        self.__x = tf.keras.layers.Dropout(0.7)(self.__x)
        self.__x = tf.keras.layers.Dense(512, activation='relu')(self.__x)
        self.__x = tf.keras.layers.Dropout(0.5)(self.__x)
        self.__x = tf.keras.layers.Dense(128, activation='relu')(self.__x)
        self.__x = tf.keras.layers.Dropout(0.3)(self.__x)
        self.__outputLayer = tf.keras.layers.Dense(13, activation='softmax')(self.__x)
        self.modelNN = tf.keras.Model(self.__inputLayer, self.__outputLayer)
        self.modelNN.compile(optimizer='Adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.modelNN.load_weights('ann/modelbestnn.2-0.3723611831665039.h5')

    # Предсказание класса новостной статьи с помощью ANN
    def predict_result(self, tokens):
        batch = 100000
        return np.argmax(self.modelNN.predict(self.batch_x_generator(tokens, batch),
                                              steps=self.__shape/batch), axis=-1)

    # Генератор порций данных для подачи методу predict нейронной сети
    @staticmethod
    def batch_x_generator(x_data, batch_size):
        samples_per_epoch = x_data.shape[0]
        number_of_batches = samples_per_epoch / batch_size
        counter = 0
        index = np.arange(np.shape(x_data)[0])
        while 1:
            index_batch = index[batch_size * counter:batch_size * (counter + 1)]
            x_batch = x_data[index_batch].toarray()
            counter += 1
            yield x_batch
            if counter > number_of_batches:
                counter = 0


# Класс LogRegClassifier, отвечающий за классификатора на основе метода логистической регрессии
class LogRegClassifier:
    # Инициализация модели
    def __init__(self):
        self.modelLR = pickle.load(open('logreg/logreg_best_alldatadop.pkl', 'rb'))

    # Предсказание класса новостной статьи c помощью метода логистической регрессии
    def predict_result(self, tokens):
        return self.modelLR.predict(tokens)


# Класс DataBase, отвечающий за работу приложения с базой данных SQLite
class DataBase:
    # Инициализация базы данных: создание (если не существует),  определение структуры таблицы и ее создание,
    # подключение к БД
    def __init__(self):
        self.__engine = create_engine('sqlite:///news.db', echo=False)
        self.table_name = 'classified_news'
        self.__metadata = MetaData(self.__engine)
        self.__tab_obj = Table(self.table_name, self.__metadata,
                               Column('id', String, primary_key=True),
                               Column('text', String, unique=True),
                               Column('newsclass', String)
                               )
        self.con = self.__engine.connect()
        self.__metadata.create_all(self.__engine, checkfirst=True)

    # Вставка данных о новой новостной статье в таблицу базы данных
    def insert_into_table(self, newstokens, newstext, newsclass):
        try:
            newsid = hashlib.md5(newstokens.encode('utf-8')).hexdigest()
            ins = self.__tab_obj.insert().prefix_with("OR IGNORE").values(id=newsid, text=newstext, newsclass=newsclass)
            self.con.execute(ins)
            return 0
        except Exception as ex:
            return ex

    # Извлечение всех данных из таблицы базы данных
    def show_tab(self):
        try:
            return self.con.execute(select([self.__tab_obj]))
        except Exception as ex:
            return ex


# Класс NewsClassifierApp, отвечающий за реализацию GUI и взаимодействие с классификатором
class NewsClassifierApp:
    # Объявление объектов пользовательского интерфейса
    def __init__(self):
        self.classifier = LogRegClassifier()
        # self.classifier = NeuralNetwork()
        self.label_encoder = pickle.load(open("preprocessing/lbl_best_alldatadop_nn.pkl", 'rb'))
        self.tfidf_transformer = pickle.load(open("preprocessing/tfidf0810__best_alldatadop_nn.pkl", 'rb'))
        self.window = Tk()
        self.menu = Menu(self.window)
        self.frameTop = Frame(self.window, bg='white', bd=5)
        self.frameBottom = Frame(self.window, bg='white', bd=5)
        self.scrollBar1 = Scrollbar(self.frameTop)
        self.textField = Text(self.frameTop, bg='white', font=20, wrap=WORD)
        self.btnPaste = Button(self.frameBottom, width=15, text='Вставить из буфера',
                               bg='white', command=self.paste_text)
        self.btnClean = Button(self.frameBottom, width=15, text='Очистить поля',
                               bg='white', command=self.delete_text)
        self.btnClassifier = Button(self.frameBottom, text='Узнать категорию', bg='white', command=self.get_class)
        self.info = Label(self.frameBottom, bg='white', font=24)
        self.frameRadio = Frame(self.window, bg='white', bd=5)
        self.r_var = BooleanVar()
        self.r1 = Radiobutton(self.frameRadio, text='Текст', bg='white', variable=self.r_var,
                              value=1, command=self.change_radio)
        self.r2 = Radiobutton(self.frameRadio, text='Файл', bg='white', variable=self.r_var,
                              value=0, command=self.change_radio)
        self.fileField = Text(self.frameRadio, bg='white', font=12, width=50, height=1, wrap='none', state=DISABLED)
        self.btnChooseFile = Button(self.frameRadio, width=15, text='Обзор',
                                    bg='white', command=self.choose_file, state=DISABLED)

    # Конфигурация объектов пользовательного интерфейса
    def config_window(self):
        self.window['bg'] = 'white'
        self.window.iconbitmap('icon.ico')
        self.window.geometry('800x800')
        self.window.resizable(width=False, height=False)
        self.window.title("Классификатор новостных статей")
        self.window.focus_set()
        self.menu.add_command(label="Открыть БД", command=self.open_db)
        self.window.config(menu=self.menu)
        font_tuple = ("Arial", 12)
        self.frameTop.place(relx=0.05, rely=0.15, relwidth=0.9, relheight=0.55)
        self.frameBottom.place(relx=0.05, rely=0.7, relwidth=0.9, relheight=0.25)
        self.scrollBar1.pack(side=RIGHT, fill=Y)
        self.textField.pack()
        self.textField.config(yscrollcommand=self.scrollBar1.set)
        self.textField.config(font=font_tuple)
        self.scrollBar1.config(command=self.textField.yview)
        self.btnPaste.pack(padx=5, pady=10, side=LEFT)
        self.btnClean.pack(padx=5, pady=10, side=RIGHT)
        self.btnClassifier.pack(padx=5, pady=10)
        self.info.pack(padx=5, pady=10)
        self.frameRadio.place(relx=0.05, rely=0.025, relwidth=0.9, relheight=0.1)
        self.r_var.set(1)
        self.r1.pack(padx=5, pady=10, side=LEFT)
        self.r2.pack(padx=5, pady=10, side=LEFT)
        self.fileField.configure(font=font_tuple)
        self.fileField.pack(padx=5, pady=10, side=LEFT)
        self.btnChooseFile.pack(padx=5, pady=10, side=RIGHT)

    # Токенизация входных данных (преобразование строки (str) текста в список токенов (list(str)))
    @staticmethod
    def razdel_tokenization(text):
        punct = list(string.punctuation)
        punct.append('«')
        punct.append('»')
        punct.append('—')
        tokens = rd.tokenize(text)
        str_tokens = []
        for token in tokens:
            if token not in punct:
                str_tokens.append(token.text)
        return str_tokens

    # Лемматизация входных данных (приведение токенов к нормальной форме)
    @staticmethod
    def lemmatization(tokens):
        ru_stopwords = stopwords.words('russian')
        morph = pymorphy2.MorphAnalyzer()
        norm_tokens = [morph.parse(token)[0].normal_form for token in tokens if token not in ru_stopwords]
        return norm_tokens

    # Векторизация текстов (преобразование токенов в числовые значения TF-IDF)
    def preprocessing_tfidf(self, text):
        text = re.sub('<[^<]+>', '', text)
        pattern = r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]' \
                  r'{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|' \
                  r'[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
        text = re.sub(pattern, ' ', text, re.IGNORECASE)
        text = str.lower(text)
        tokens = self.razdel_tokenization(text)
        tokens = self.lemmatization(tokens)
        tokens = [t for t in tokens if re.match('[a-zа-я]+-?[a-zа-я]+', t)]
        tftokens = self.tfidf_transformer.transform([tokens])
        return tftokens, tokens

    # Обработчик события нажатия на кнопку "Узнать категорию" - определение класса новостной статьи
    def get_class(self):
        try:
            if self.textField.get(1.0, 'end') != '\n':
                db = DataBase()
                text = self.textField.get("1.0", "end")
                tokens, words = self.preprocessing_tfidf(text)
                predictions = self.classifier.predict_result(tokens)
                label = self.label_encoder.inverse_transform([int(predictions[0])])[0]
                self.info['text'] = label
                # Если класс новости определен успешно, производится вставка данных в таблицу БД
                state = db.insert_into_table(' '.join(words), text, label)
                if state != 0:
                    raise Exception(state)
                db.con.close()
            else:
                raise ValueError
        except ValueError:
            mb.showerror(title='Ошибка текстовых данных', message='Введен некорректный текст. Повторите попытку.')
        except Exception as ex:
            mb.showerror(title='Ошибка базы данных', message=ex.args[0])

    # Обработчик события нажатия на кнопку "Очистить поля" - удаление данных из всех текстовых полей
    def delete_text(self):
        self.textField.delete(1.0, "end")
        self.info['text'] = ''
        self.fileField['state'] = NORMAL
        self.fileField.delete(1.0, "end")
        self.fileField['state'] = DISABLED
        if self.r_var.get() == 0:
            self.textField['state'] = DISABLED

    # Обработчик события нажатия на кнопку "Вставить из буфера" - извлечение текста из буфера обмена
    # и добавление в основное текстовое поле
    def paste_text(self):
        try:
            if self.r_var.get() == 1:
                self.textField['state'] = NORMAL
                self.textField.delete(1.0, "end")
                self.textField.insert(1.0, pc.paste())
        except Exception as ex:
            mb.showerror(title='Ошибка', message=ex)

    # Обработчик события нажатия на кнопку "Обзор" - открытие диалогового окна выбора текстового
    # файла для извлечения текста новостной статьи и его добавления в основное текстовое поле
    def choose_file(self):
        try:
            filetypes = [("Текстовый файл", "*.txt")]
            filename = fd.askopenfilename(title="Открыть файл", filetypes=filetypes)
            if filename != '':
                self.fileField['state'] = NORMAL
                self.fileField.delete(1.0, 'end')
                self.fileField.insert(1.0, str(filename))
                self.fileField['state'] = DISABLED
                self.textField['state'] = NORMAL
                with open(filename, 'r', encoding='utf-8') as reader:
                    self.textField.delete(1.0, 'end')
                    self.textField.insert(1.0, reader.read())
        except Exception as ex:
            mb.showerror(title='Ошибка', message=ex)

    # Обработчик события изменения значения переключателей - блокировка полей
    def change_radio(self):
        self.delete_text()
        if self.r_var.get() == 1:
            self.btnChooseFile['state'] = DISABLED
            self.textField['state'] = NORMAL
        else:
            self.btnChooseFile['state'] = NORMAL
            self.textField['state'] = DISABLED
        self.delete_text()

    # Сохранение всех данных таблицы БД в CSV-файл
    @staticmethod
    def save_csv():
        try:
            db = DataBase()
            content = db.show_tab()
            filetypes = [("CSV-файл", "*.csv")]
            file = fd.asksaveasfile(mode='w', initialdir="/", filetypes=filetypes)
            csv_out = csv.writer(file)
            csv_out.writerow(('id', 'text', 'class'))
            for row in content:
                csv_out.writerow((row[0], re.sub('\n', '', row[1]), row[2]))
            file.close()
            db.con.close()
            mb.showinfo('Сохранение таблицы', message='Таблица успешно записана в файл ' + file.name)
        except Exception as ex:
            mb.showerror(title='Ошибка', message=ex)

    # Обработчик события нажатия на кнопку меню "Открыть БД" - отображение таблицы БД
    # с классифицированными новостными статьями
    def open_db(self):
        db_view = Toplevel(self.window)
        db_view.geometry('900x600')
        db_view.iconbitmap('icon.ico')
        db_view.resizable(width=False, height=False)
        db_view.title("Таблица classified_news")
        db_view.focus_set()
        db_view.grab_set()
        menu2 = Menu(self.window)
        menu2.add_command(label="Сохранить в CSV-файл", command=self.save_csv)
        db_view.config(menu=menu2)
        style = ttk.Style(db_view)
        style.configure('Treeview', rowheight=90)
        columns = ("id", "text", "class")
        table = ttk.Treeview(db_view, show="headings",
                             columns=columns, height=13)
        table.heading("id", text="ID")
        table.column("id", width=100)
        table.heading("text", text="Текст")
        table.column("text", width=650)
        table.heading("class", text="Класс")
        table.column("class", width=150)
        scrollbar3 = Scrollbar(db_view)
        scrollbar3.pack(side=RIGHT, fill=Y)
        table.configure(yscroll=scrollbar3.set)
        scrollbar3.config(command=table.yview)
        table.pack()
        db = DataBase()
        content = db.show_tab()
        for row in content:
            table.insert("", END, values=(row[0], row[1], row[2]), )
        db.con.close()


# Точка входа в программу
if __name__ == "__main__":
    # Предотвращение загрузки нейронной сети в память GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    nca = NewsClassifierApp()
    nca.config_window()
    nca.window.mainloop()
