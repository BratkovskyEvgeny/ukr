# импортируем библиотеки pandas, numpy
import pandas as pd
import numpy as np
# импортируем библиотеку streamlit
import streamlit as st
# импортируем модуль dill
import dill


hide_streamlit_style = """  
<style>

footer {visibility: hidden;}
#sidebar {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)





# функция предварительной подготовки
def preprocessing(df):

    # значения переменной age меньше 18 заменяем
    # минимально допустимым значением возраста
    df['age'] = np.where(df['age'] < 18, 18, df['age'])

    # создаем переменную Ratio - отношение количества
    # просрочек 90+ к общему количеству просрочек
    sum_of_delinq = (df['NumberOfTimes90DaysLate'] +
                     df['NumberOfTime30-59DaysPastDueNotWorse'] +
                     df['NumberOfTime60-89DaysPastDueNotWorse'])

    cond = (df['NumberOfTimes90DaysLate'] == 0) | (sum_of_delinq == 0)
    df['Ratio'] = np.where(
        cond, 0, df['NumberOfTimes90DaysLate'] / sum_of_delinq)

    # создаем индикатор нулевых значений переменной
    # NumberOfOpenCreditLinesAndLoans
    df['NumberOfOpenCreditLinesAndLoans_is_0'] = np.where(
        df['NumberOfOpenCreditLinesAndLoans'] == 0, 'T', 'F')

    # создаем индикатор нулевых значений переменной
    # NumberRealEstateLoansOrLines
    df['NumberRealEstateLoansOrLines_is_0'] = np.where(
        df['NumberRealEstateLoansOrLines'] == 0, 'T', 'F')

    # создаем индикатор нулевых значений переменной
    # RevolvingUtilizationOfUnsecuredLines
    df['RevolvingUtilizationOfUnsecuredLines_is_0'] = np.where(
        df['RevolvingUtilizationOfUnsecuredLines'] == 0, 'T', 'F')

    # преобразовываем переменные в категориальные, применив
    # биннинг и перевод в единый строковый формат
    for col in ['NumberOfTime30-59DaysPastDueNotWorse',
                'NumberOfTime60-89DaysPastDueNotWorse',
                'NumberOfTimes90DaysLate']:
        df.loc[df[col] > 3, col] = 4
        df[col] = df[col].apply(lambda x: f"cat_{x}")

    # создаем список списков - список 2-факторных взаимодействий
    lst = [
        ['NumberOfDependents',
         'NumberOfTime30-59DaysPastDueNotWorse'],
        ['NumberOfTime60-89DaysPastDueNotWorse',
         'NumberOfTimes90DaysLate'],
        ['NumberOfTime30-59DaysPastDueNotWorse',
         'NumberOfTime60-89DaysPastDueNotWorse'],
        ['NumberRealEstateLoansOrLines_is_0',
         'NumberOfTimes90DaysLate'],
        ['NumberOfOpenCreditLinesAndLoans_is_0',
         'NumberOfTimes90DaysLate']
    ]

    # создаем взаимодействия
    for i in lst:
        f1 = i[0]
        f2 = i[1]
        df[f1 + ' + ' + f2 + '_interact'] = (df[f1].astype(str) + ' + '
                                             + df[f2].astype(str))
    return df

# загрузим сохраненную ранее модель
with open('pipeline_for_deployment.pkl', 'rb') as f:
    pipe = dill.load(f)

#  функция запуска web-интерфейса
def run():
    from PIL import Image
    image = Image.open('logo.png')
    

    st.sidebar.image(image)

    sidebar_ttl = ("Приложение для нетворкинга с УКР")
    st.sidebar.info(sidebar_ttl)
    
    question = ("В каком режиме сделать прогноз, Онлайн\n"
                "(Online) или загрузкой файла данных(Batch)?")
    
    add_selectbox = st.sidebar.selectbox(question, ("Online", "Batch"))

    st.sidebar.info("Разработчик - Братковский Е.В.")
                   
   

    st.title("Прогнозирование дефолта🏦💰")
    st.write("Демка для Валентина, Станислава, Тимофея, Павла)

    if add_selectbox == "Online":
        
        RevolvingUtilizationOfUnsecuredLines = \
            st.number_input("RevolvingUtilizationOfUnsecuredLines")
        age = st.number_input("age", step=1)
        NumberOfTime30_59DaysPastDueNotWorse = \
            st.number_input("NumberOfTime30-59DaysPastDueNotWorse",
                            step=1)
        DebtRatio = \
            st.number_input("DebtRatio")
        MonthlyIncome = \
            st.number_input("MonthlyIncome")
        NumberOfOpenCreditLinesAndLoans = \
            st.number_input("NumberOfOpenCreditLinesAndLoans", step=1)
        NumberOfTimes90DaysLate = \
            st.number_input("NumberOfTimes90DaysLate", step=1)
        NumberRealEstateLoansOrLines = \
            st.number_input("NumberRealEstateLoansOrLines", step=1)
        NumberOfTime60_89DaysPastDueNotWorse = \
            st.number_input("NumberOfTime60-89DaysPastDueNotWorse", step=1)
        NumberOfDependents = st.number_input("NumberOfDependents", step=1)
        output = ""

        input_dict = {
            'RevolvingUtilizationOfUnsecuredLines':
            RevolvingUtilizationOfUnsecuredLines,
            'age': age,
            'NumberOfTime30-59DaysPastDueNotWorse': 
            NumberOfTime30_59DaysPastDueNotWorse,
            'DebtRatio': DebtRatio,
            'MonthlyIncome': MonthlyIncome,
            'NumberOfOpenCreditLinesAndLoans': 
            NumberOfOpenCreditLinesAndLoans,
            'NumberOfTimes90DaysLate': 
            NumberOfTimes90DaysLate,
            'NumberRealEstateLoansOrLines': 
            NumberRealEstateLoansOrLines,
            'NumberOfTime60-89DaysPastDueNotWorse': 
            NumberOfTime60_89DaysPastDueNotWorse,
            'NumberOfDependents': NumberOfDependents
        }
        input_df = pd.DataFrame([input_dict])

        if st.button("Спрогнозировать вероятность дефолта"):

            # выполняем предварительную обработку новых данных
            input_df = preprocessing(input_df)

            # вычисляем вероятности для новых данных
            output = pipe.predict_proba(input_df)[:, 1]
            output = str(output)

        st.success("Вероятность дефолта: {}".format(output))

    if add_selectbox == "Batch":
        st.balloons()
        
        file_upload_ttl = ("Загрузите csv-файл с новыми данными\n"
                           "для вычисления вероятностей:")
        file_upload = st.file_uploader(file_upload_ttl, type=['csv'])

        if file_upload is not None:
            newdata = pd.read_csv(file_upload)
            # выполняем предварительную обработку новых данных
            newdata = preprocessing(newdata)

            # вычисляем вероятности для новых данных
            prob = pipe.predict_proba(newdata)[:, 1]

            # вывод вероятностей на веб-странице
            st.success("Вероятности дефолта для загруженных данных:")
            st.write(prob)


if __name__ == '__main__':
    run()
