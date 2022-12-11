1. Положить `train`, `dev`, `test` папки `NEREL` датасета в `data/nerel` директорию.

2. Запустить обучение: `CUDA_VISIBLE_DEVICES=0 python train.py --bert_model=Geotrend/bert-base-10lang-cased --dataset_name=nerel --output_dir=out --per_device_train_batch_size=8 --per_device_eval_batch_size=16 --evaluation_strategy=steps --eval_steps=100 --logging_steps=100 --span_coef=1.0 --start_coef=0.0 --end_coef=0.0 --num_train_epochs=3 --learning_rate=5e-5 --max_sequence_length=512 --save_path=model_3ep.pkl`

3. Квантизовать: `python optimize.py --model model_3ep.pkl --quantize=True --fuse=True`

4. Оценить качество на тесте: `python solution.py`

Основная идея: переход к классификации спанов с помощью эмбеддинга длины + предсказывать принадлежность спана к сущности с помощью косинусной схожести представления этого спана с представлением, полученным из описания этого типа сущности.
