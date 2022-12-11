import json

if __name__ == '__main__':
    with open('data/dev-dataset-task2022-04.json') as f:
        data = json.load(f)

    preprocessed_data = []
    for example_text, example_category in data:
        if not len(example_text):
            continue

        if 'category' in example_text:
            print('DELETED: ' + example_text)
            continue

        if "Drag'n'drop" in example_text:
            print('DELETED: ' + example_text)
            continue

        if 'javascript' in example_text or 'JavaScript' in example_text and len(example_text) < 200:
            print('DELETED: ' + example_text)
            continue

        if example_text == 'Печать Нашли ошибку в тексте? Выделите ее и нажмите Ctrl + Enter':
            print('DELETED: ' + example_text)
            continue

        if example_text == 'Чтобы сообщить нам об опечатке, выделите ее мышкой и нажмите Ctrl+Enter':
            print('DELETED: ' + example_text)
            continue

        if example_text == 'Лента новостей Лента новостей':
            print('DELETED: ' + example_text)
            continue

        if example_text == 'В мире Добавил: Oksana Сегодня, 02:00':
            print('DELETED: ' + example_text)
            continue

        if example_text == 'Tab 2 content goes here... Tab 3 content goes here...':
            print('DELETED: ' + example_text)
            continue

        if example_text == 'Криштиану Роналду / Фото: © Michael Regan / Staff / Getty Images Sport / Gettyimages.ru':
            print('DELETED: ' + example_text)
            continue

        if example_text == 'Мы говорим вам правду. Вы решаете, что с ней делать.':
            print('DELETED: ' + example_text)
            continue

        if example_text == '× Вы можете редактировать свой комментарий только в течении 5 минут':
            print('DELETED: ' + example_text)
            continue

        if example_text == 'Вы собираетесь перейти по внешней ссылке: Вы действительно хотите перейти по ссылке?':
            print('DELETED: ' + example_text)
            continue

        if example_text == 'Oh boy!':
            print('DELETED: ' + example_text)
            continue

        if example_text == 'Авто Добавил: tantan61 Вчера, 13:30':
            print('DELETED: ' + example_text)
            continue

        if example_text == '24.04.18 7:38 текст: Ирина Клячина фото: скриншот Яндекс.Картинки 49':
            print('DELETED: ' + example_text)
            continue

        preprocessed_data.append((example_text, example_category))

    # deduplicate
    preprocessed_data = set(preprocessed_data)

    with open('data/dev-dataset-task2022-04_preprocessed.json', 'w') as f:
        json.dump(list(preprocessed_data), f, ensure_ascii=False, indent=4)
