
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime, date
import itertools
from ast import literal_eval

from skllm.config import SKLLMConfig
from skllm.models.gpt.text2text.summarization import GPTSummarizer
from skllm.models.gpt.classification.zero_shot import MultiLabelZeroShotGPTClassifier
import openai
from openai import OpenAI

nltk.download('punkt')
nltk.download('stopwords')
st.set_page_config(layout='wide')

api_key = st.secrets["api_key"]#open('openaiapikey.txt').read()
client = OpenAI(api_key=api_key)
SKLLMConfig.set_openai_key(api_key)
df = pd.read_csv("data/Philippine Fake News Corpus - cleaned.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Start of Month'] = (df['Date'].dt.to_period('M').dt.to_timestamp()).dt.date
df['Start of Week'] = (df['Date'] - pd.to_timedelta(df['Date'].dt.dayofweek, unit='D')).dt.date
df['Date'] = df['Date'].dt.date
df['Authors'] = df['Authors'].apply(literal_eval)

source_author_df = df[['Source', 'Authors']]
source_author_df  = source_author_df.explode('Authors')
source_author_df = source_author_df.drop_duplicates()
sources = df['Source'].unique()

source_author_df = source_author_df.sort_values(by = ['Authors'], ascending=True, na_position='last')
source_author_df['Authors'] = source_author_df['Authors'].fillna(source_author_df['Source'] +" (No Author)")
authors = source_author_df['Authors'].unique()

df['Authors_'] = df.apply(lambda x: [x['Source'] +" (No Author)"] if x['Authors'] == [] else x['Authors'] , axis=1)

date_min = df['Date'].min()
date_max = df['Date'].max()

#helper functions


def article_count_num(df,label_suffix):
    html = f'''
        <div style="background-color: transparent; padding: 20px 5px 5px 5px; text-align: center; justify: center; height:600">
            <div style="font-size: 3rem; font-weight: bold; margin-bottom: 0px; margin-top: 20px;">{df['Headline'].count()}</div>
            <div style="font-size: 1.5rem; font-weight: bold; margin-bottom: 0px;">{label_suffix}News Articles</div>
            <div style="font-size: 1.1rem; ">from {df['Date'].min().strftime("%B %d, %Y")}</div>
            <div style="font-size: 1.1rem; ">to {df['Date'].max().strftime("%B %d, %Y")}</div>
    
        </div>
        '''
    return html
    

def has_intersection(names, target_names):
    return bool(set(names) & set(target_names))
    
def article_count_period(df, period, period_column, label_suffix,line_color ):
    df_period_count = df.groupby([period_column])['Headline'].count().to_frame('article_count').reset_index()
    chart = alt.Chart(df_period_count).mark_line(
            color=line_color,
            opacity=1
            ).encode(
            x=alt.X(period_column),
            y=alt.Y('article_count'),
            tooltip=[period_column, 'article_count']
            ).properties(
                title=f'{period} Number of {label_suffix}News Articles',
  
                height=300
            ).configure_axis(
                # labelFontSize=0,  # Set label font size to 0 to hide axis labels
                title=None  # Set axis title to None to hide axis title
            ).configure_title(
                fontSize=20,  # Adjust title font size
                # anchor='start',  # Title alignment
                # color='blue'  # Title color
            )
    return chart

def source_count(df,bar_color,label_suffix):
    st.header(f':writing_hand: Top Sources for {label_suffix}News Dataset')
    df_source_count = df.groupby(['Source']).size().to_frame('article_count').reset_index()
    df_source_count.sort_values(by = 'article_count', ascending = False)
    
    chart = alt.Chart(df_source_count).mark_bar(color = bar_color).encode(
        x= 'article_count:Q',
        y=alt.Y('Source:N', sort='-x'),
        tooltip=['Source:N', 'article_count:Q'] 
    ).properties(
        # title = f"{label_suffix}News Sources",
        width=600,
        height=alt.Step(40)
    ).configure_axis(
        # labelFontSize=0,  # Set label font size to 0 to hide axis labels
        title=None  # Set axis title to None to hide axis title
    ).configure_title(
        fontSize=20,  # Adjust title font size
        # anchor='start',  # Title alignment
        # color='blue'  # Title color
    )

    return chart

def author_count(df,bar_color,label_suffix,top_n):
    st.header(f':writing_hand: Top Authors for {label_suffix}News Dataset')
    df_author_count = df.explode('Authors')
    df_author_count['Author'] = df_author_count['Authors'].fillna(df_author_count['Source'] +" (No Author)")
    df_author_count = df_author_count.drop('Authors', axis = 1)
    df_author_count = df_author_count.groupby(['Author']).size().to_frame('article_count')
    df_author_count.sort_values(by = 'article_count', ascending = False)
    df_author_count = df_author_count.reset_index()
    df_author_count = df_author_count.tail(top_n)
    
    chart = alt.Chart(df_author_count).mark_bar(color = bar_color).encode(
        x= 'article_count:Q',
        y=alt.Y('Author:N', sort='-x'),
        tooltip=['Author:N', 'article_count:Q'] 
    ).properties(
        # title = f"Top {label_suffix}News Authors",
        width=600,
        height=alt.Step(40)
    ).configure_axis(
        # labelFontSize=0,  # Set label font size to 0 to hide axis labels
        title=None  # Set axis title to None to hide axis title
    ).configure_title(
        fontSize=20,  # Adjust title font size
        # anchor='start',  # Title alignment
        # color='blue'  # Title color
    )

    return chart

def tokenize_contents(df):
    content = df['Content'].str.cat(sep=' ')
    tokens = word_tokenize(content)
    tokens = [word.lower() for word in tokens
              if word not in stopwords.words('english')
              and word.isalpha()]
    return content,tokens
    
def top_bigrams(tokens,label_suffix, top_n,bar_color):
    st.header(f":bar_chart: Top bigrams from the {label_suffix}News Dataset")

    bigrams = list(nltk.bigrams(tokens))
    bigram_counts = nltk.FreqDist(bigrams)
    top_10_bigrams = bigram_counts.most_common(top_n)

    bigram_words = [f"{word1} {word2}" for (word1, word2), freq in top_10_bigrams]
    bigram_frequencies = [freq for (word1, word2), freq in top_10_bigrams]

    df_bigrams = pd.DataFrame({'Bigrams': bigram_words, 'Frequency': bigram_frequencies})
    chart = alt.Chart(df_bigrams).mark_bar(color = bar_color).encode(
        x= 'Frequency:Q',
        y=alt.Y('Bigrams:N', sort='-x'),
        tooltip=['Bigrams:N', 'Frequency:Q'] 
    ).properties(
        # title = f"Top {label_suffix}News Authors",
        width=600,
        height=alt.Step(40)
    ).configure_axis(
        # labelFontSize=0,  # Set label font size to 0 to hide axis labels
        title=None  # Set axis title to None to hide axis title
    ).configure_title(
        fontSize=20,  # Adjust title font size
        # anchor='start',  # Title alignment
        # color='blue'  # Title color
    )

    return chart

def word_cloud(content,label_suffix, color = None):
    st.header(f':capital_abcd: Wordcloud from the {label_suffix}News Dataset')

    stop_words=stopwords.words('english')
    wordcloud = WordCloud(width = 1000, height = 600,
                          background_color ='white',
                          stopwords = stop_words,
                          min_font_size = 10).generate(content)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)        
    return fig
        



# Function to label an article
def label_article(Headline):
    clf = MultiLabelZeroShotGPTClassifier(openai_model="gpt-3.5-turbo", max_labels=3)    
    # Define the candidate labels
    candidate_labels = [
        'Politics',
        'Economy',
        'Business',
        'Technology',
        'Health',
        'Environment',
        'Education',
        'Entertainment',
        'Sports',
        'International Relations',
        'Lifestyle',
    ] 
    clf.fit(None, [candidate_labels])
    return clf.predict([Headline])[0]


def extract_keywords(text):
    system_prompt = 'You are a news analyst assistant tasked to extract keywords from news articles.'

    main_prompt = """
    ###TASK###
    - Extract the five most crucial keywords from the news article. 
    - Extracted keywords must be listed in a comma-separated list. 
    - Example: digital advancements, human rights, AI, gender, post-pandemic

    ###ARTICLE###
    """

    try:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo', 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{main_prompt}\n{text}"}
            ]
        )
        top_keywords = response.choices[0].message.content
        return [kw.strip() for kw in top_keywords.split(',')]

    except e:        
        return []

def generate_response(article, prompt):
    response = client.chat.completions.create(
        model='gpt-3.5-turbo', 
        messages=[
            {'role': 'system', 
             'content': 
             f"Perform the specified tasks based on this article:\n\n{article}"},
            {'role': 'user', 'content': prompt}
        ]
    )
    return response.choices[0].message.content

def identify_entities(article):
    prompt = f"""
    Return a list of persons and organization entities mentioned in the article
    """
        #     - Entities must be listed in a comma-separated list
        # - Return the list only. 
        # - Example: Elon Musk,Rodrigo Duterte,PhilHealth

    entities = generate_response(article, prompt)
    return entities

def identify_entity_sentiment(article, entities):
    prompt = f"""
    Identify the article's sentiment on each mentioned entity whether POSTITIVE, NEUTRAL or NEGATIVE:\n\n{entities}.        
    """
        # - List them in a semi-colon-separated format along with their sentiment, POSTITIVE, NEUTRAL or NEGATIVE. 
        # - Return List Only
        # - Example: Elon Musk,POSITIVE; Rodrigo Duterte,NEGATIVE; PhilHealth,'NEUTRAL'
    
    
    sentiment = generate_response(article, prompt)
    return sentiment

def support_claims(article, entity_sentiment):
    prompt = f""""
        f"For each entity and sentiment in {entity_sentiment}, identify claims that support the sentiment on entity"   
    """
    # st.write(1)
    descriptions = generate_response(article, prompt)
    return descriptions


    
def identify_entities_with_sentiment(article):
    prompt = f"Identify only the person and organization entities mentioned in the article along with the article's sentiment on each mentioned entity (whether Positive, Neutral or Negative)."
    entity = generate_response(article, prompt)
    return entity
    
############SET COMMON FILTERS#########
# filter_default_df = pd.read_csv("data/filter_default.csv")
# st.write(filter_default_df)
# st.write(filter_default_df)

def initialize_session_state():
    st.session_state.label = 'Not Credible'
    st.session_state.date_filter = True
    st.session_state.start_date = date_min
    st.session_state.end_date = date_max
    st.session_state.source_filter = True
    st.session_state.source = []
    st.session_state.author_filter = True
    st.session_state.author = []
    st.session_state.keyword_filter = True
    st.session_state.keywords = ''
    st.session_state.search_cols = []
    st.session_state.df_filtered = pd.DataFrame(columns = ['Headline'])
    st.session_state.show_common_filter_interactive_highlights = True
    st.session_state.show_common_filter_summarization = True
    st.session_state.filter_article_summarization = False
    st.session_state.show_common_filter_ml_classifier = True
    st.session_state.filter_article_ml_classifier = False
    st.session_state.show_common_filter_interactive_quiz = True
    st.session_state.filter_article_interactive_quiz = False
    st.session_state.article = None
    st.session_state.headline = None
    st.session_state.headline_index = None

def get_session_state():
    session_state = st.session_state
    if 'start_date' not in session_state:
        initialize_session_state()
    return session_state
    
def common_filter():
    session_state = get_session_state()
    frow1_col1, frow1_col2, frow1_col3 , frow1_col4 , frow1_col5, frow1_col6= st.columns([2.5,1,1,2,0.5,2])
    frow2_col1, frow2_col2= st.columns([1,8])
    frow3_col1, frow3_col2= st.columns([1,8])
    frow4_col1, frow4_col2= st.columns([1,8])
    frow5_col1, frow5_col2, frow5_col3= st.columns([7,1,1])

    labels = ['Credible', 'Not Credible', 'All']
    label_temp = frow1_col1.selectbox('Label:', labels, index=labels.index(session_state.label))
    frow1_col3.write("<div style='height: 30px;font-size: 14px'>Filter Date:</div>", unsafe_allow_html=True)
    date_filter_temp = frow1_col3.toggle('', value=session_state.date_filter , key = 'date_filter_key')
    if date_filter_temp == True:
        start_date_temp = frow1_col4.date_input('',
                              value = session_state.start_date,
                              min_value=date_min,
                              max_value=date_max,
                              # key = 'start_date'
                              )
        frow1_col5.write("<div style='padding-top:35px; text-align:center;font-size: 14px' > to </div>", unsafe_allow_html=True)
        end_date_temp = frow1_col6.date_input('',
                              value = session_state.end_date,
                              min_value=session_state.start_date,
                              max_value=date_max,
                              # key = 'end_date'
                            )

        
    frow2_col1.write("<div style='height: 30px;font-size: 14px'>Filter Source:</div>", unsafe_allow_html=True)
    source_filter_temp = frow2_col1.toggle('', value=session_state.source_filter, key = 'source_filter_key')
    
    if source_filter_temp == True:
        for s in  session_state.source:
            if s not in sources:
                session_state.source.remove(s)
        source_temp = frow2_col2.multiselect('', sources, session_state.source)

    frow3_col1.write("<div style='height: 30px;font-size: 14px'>Filter Author:</div>", unsafe_allow_html=True)    
    author_filter_temp = frow3_col1.toggle('', value=session_state.author_filter, key = 'author_filter_key')
    if author_filter_temp == True:
        author_temp = frow3_col2.multiselect('', authors, session_state.author)

    frow4_col1.write("<div style='font-size: 14px'>Filter Keywords:</div>", unsafe_allow_html=True)    
    keyword_filter_temp = frow4_col1.toggle('', value=session_state.keyword_filter, key = 'keyword_filter_key')
    
    if keyword_filter_temp == True:        
        keywords_temp = frow4_col2.text_input(
            label='Keywords for filtering the data. If multiple keywords, make a comma-separated list',
            value=session_state.keywords
        )
        
        search_cols_temp = frow4_col2.multiselect('Select columns where keywords will be searched',
                            ['Headline', 'Content', 'URL'], session_state.search_cols)
    
    # if frow5_col2.button("Clear", key='clear-button', type="secondary", use_container_width=True):
    #     session_state.source.clear()
    #     session_state.author.clear()
    #     session_state.search_cols.clear()
    #     session_state.keywords = ''
    #     st.experimental_rerun()    
    
    if frow5_col3.button("Apply", key='apply-button', type="primary", use_container_width=True):
        session_state.label = label_temp
        session_state.date_filter = date_filter_temp
        if date_filter_temp:
            session_state.start_date = start_date_temp
            session_state.end_date = end_date_temp
        
        session_state.source_filter = source_filter_temp
        if source_filter_temp:
            session_state.source = source_temp
        
        session_state.author_filter = author_filter_temp
        if author_filter_temp:
            session_state.author = author_temp
        
        session_state.keyword_filter = keyword_filter_temp
        if keyword_filter_temp:
            session_state.keywords = keywords_temp 
            session_state.search_cols = search_cols_temp 
        
        if session_state.label == 'All':
            df_filtered = df
        else:
            df_filtered = df[df['Label'] == session_state.label]

        if session_state.date_filter == True and len(df_filtered)>0:
            df_filtered = df_filtered[(df_filtered['Date'] >= session_state.start_date) & (df_filtered['Date'] <= session_state.end_date) ]            

        if session_state.source_filter == True and len(df_filtered)>0:
            df_filtered = df_filtered[df_filtered['Source'].isin(session_state.source)]
        
        if session_state.author_filter== True and len(df_filtered)>0:
            df_filtered = df_filtered[df_filtered['Authors_'].apply(lambda x: has_intersection(x, session_state.author))]

        if df_filtered.shape[0] > 0 and len(df_filtered)>0:
            keywords_list = [kw.strip() for kw in session_state.keywords.split(',')] 
            
            if session_state.search_cols:
                marker_cols = list()
                for col in session_state.search_cols:
                    df_filtered[col+'_marker'] = df_filtered[col].str.contains('|'.join(keywords_list), case=False)               
                    marker_cols.append(col+'_marker')
        
                search_markers = [x + '_marker' for x in session_state.search_cols]
                df_filtered['marker'] = df_filtered[search_markers].sum(axis=1)
                df_filtered['marker'] = df_filtered['marker'] > 0
        
                df_filtered = df_filtered[df_filtered['marker']]
                
        if len(df_filtered)>0:
            df_filtered.reset_index(inplace = True, drop = True)
        session_state.df_filtered = df_filtered  
        
    return session_state.df_filtered

    
# Page 1 content
def about_the_data():
    st.title(":mag_right: A Philippine Fake News Exploration App")
    st.markdown("This Streamlit app is a response to the campaign against [misinformation and disinformation](https://www.un.org/en/countering-disinformation) in the Philippines.")
    st.markdown("It offers an in-depth analysis of the Philippine Fake News Corpus, which comprises web-scraped news articles from January 1, 2016, to October 31, 2018. For this prototype app, only the articles published from 2018 will be used. For more information about the dataset, you may access [this link](https://github.com/aaroncarlfernandez/Philippine-Fake-News-Corpus/tree/master).")

    st.header("Preview of the dataset")
    st.write(df.head())
    st.header("Quick stats from Rappler news articles")

    col1, col2 =st.columns([2,5])
    col1.write(article_count_num(df[['Headline', 'Content', 'Date', 'Source', 'Authors', 'URL', 'Label']],''), unsafe_allow_html=True)    
    col2.altair_chart(article_count_period(df, 'Monthly' ,'Start of Month', '', '#1E90FF'), use_container_width=True)
    
# Page 2 content


    
def interactive_highlights():
    session_state = get_session_state()
    st.title(':newspaper: Interacting with the 2018 Philippine Fake News Corpus')
    show_common_filter = st.toggle('Show Filter', value=True , key = 'show_common_filter_interactive_key')
    if show_common_filter == True:
        df_filtered = common_filter()
    else:
        df_filtered = session_state.df_filtered
        
    if st.session_state.label == 'Credible':
        label_suffix = 'Credible '
        color =  'green'#'#33FF57'
    elif st.session_state.label == 'Not Credible':  
        label_suffix = 'Fake '
        color = 'red'#'#FF5733'
    else:
        label_suffix = ''
        color =  '#1E90FF'
    
    if df_filtered.shape[0]>0:            
        viz_col1,viz_col2 = st.columns([2,5])
        period = viz_col2.selectbox('Date Column:', ['Daily', 'Weekly', 'Monthly'], index=1)
        if period == 'Daily':
            date_column = 'Date'
        elif period == 'Weekly':
            date_column = 'Start of Week'
        elif period == 'Monthly':
            date_column = 'Start of Month'

        viz_col1.write(article_count_num(df_filtered,label_suffix), unsafe_allow_html=True)    
        viz_col2.altair_chart(article_count_period(df_filtered, period , date_column, label_suffix, color ), use_container_width=True)
        st.altair_chart(source_count(df_filtered, color , label_suffix), use_container_width=True)
        st.altair_chart(author_count(df_filtered,'grey',label_suffix,top_n = 10), use_container_width=True)
        
        content, tokens = tokenize_contents(df_filtered)
        st.altair_chart(top_bigrams(tokens,label_suffix,10,color), use_container_width=True)

        word_cloud_fig = word_cloud(content,label_suffix) 
        st.pyplot(word_cloud_fig)
        
        show_filtered_data = st.toggle('Show Filter Data', value=False , key = 'show_filtered_data_key')               
        if show_filtered_data:
            st.write(df_filtered[['Headline', 'Content', 'Date', 'Source', 'Authors', 'URL', 'Label']])
        
    else:
        st.write("<div style='height: 80px;font-size: 30px;text-align: center; padding:20px'>No Found Article</div>", unsafe_allow_html=True)   

def news_summarization():
    st.title('Summarizing 2018 Articles')
    session_state = get_session_state()
    
    sf_col1, sf_col2, sf_col3 = st.columns([2,2,5])
    session_state = get_session_state()
    filter_headlines = sf_col1.toggle('Filter Articles', value=session_state.filter_article_summarization, key = 'filter_article_summarization_key')
    df_filtered = df 

    if filter_headlines == True:
        show_common_filter= sf_col2.toggle('Show Filter', value=False, key = 'show_common_filter_summarization_key')    
        
        if show_common_filter == True:
             common_filter()        
        df_filtered = session_state.df_filtered

    headline_temp = None
    if len(df_filtered) >0:
        headlines = df_filtered['Headline'].to_list()
        art_col1, art_col2 = st.columns([7,2])
        
        headline_temp = art_col1.selectbox('Select article title', headlines , index=session_state.headline_index)
        
        art_col2.write("<div style='height:30px'> </div>", unsafe_allow_html=True)           
        apply_article = art_col2.button('Apply',use_container_width = True, key = 'article_apply_key')
       
        if apply_article:
            session_state.headline = headline_temp            
            if headline_temp is None:
                session_state.headline_index = None
                session_state.article = None
            else:
                session_state.headline_index = headlines.index(headline_temp)
                article_temp = df_filtered[df_filtered['Headline']==headline_temp].iloc[0]
                session_state.article = article_temp       

            
        if apply_article and session_state.article is not None:
            article = session_state.article 
            st.header(f"[{article['Headline']}]({article['URL']})")
            st.caption(f"__Published date:__ {article['Date']}")
                        # Predict labels           
            predicted_labels = label_article(article['Content'])
            # Display predicted labels as blue tabs
            st.caption('**PREDICTED LABELS**')
            labeled_categories = ""
            for label in predicted_labels:
                if label!= "":
                    labeled_categories += f"<span style='background-color:#0041C2;padding: 5px; border-radius: 5px; margin-right: 5px;'>{label}</span>"
            st.markdown(labeled_categories, unsafe_allow_html=True)
    
            
            top_keywords = extract_keywords(article['Content'])
            if len(top_keywords) >0:
                st.caption('**TOP KEYWORDS**')  
            highlighted_keywords = ""
            for i, keyword in enumerate(top_keywords):
                highlighted_keywords += f"<span style='background-color:#0041C2;padding: 5px; border-radius: 5px; margin-right: 5px;'>{keyword}</span>"
    
            st.markdown(highlighted_keywords, unsafe_allow_html=True)    
            st.caption('**ENTITIES WITH SENTIMENTS**')
            try:
                st.subheader("Article's sentiment on entities mentioned")                     
                identify_entities2 = identify_entities_with_sentiment(article)
                st.write(identify_entities2)
            except:
                 pass
            # try:
            # entities = identify_entities(article['Content'])
            # entity_sentiment = identify_entity_sentiment(article['Content'], entities)
            # entity_sentiment_claims = support_claims(article['Content'], entity_sentiment)
            # st.write(entity_sentiment_claims)
            
            # entities_ = [ent.strip() for ent in entities.split(',')]
             
            # entities_positive_list = []
            # entities_negative_list = []
            # entities_neutral_list = []
            # st.write(entity_sentiment)
            # for ent_sen in entity_sentiment.split(';') :
            #     items = ent_sen.split(',')
            #     ent = ','.join(items[:-1])
            #     sen = items[-1]
            #     if sen == 'POSITIVE':
            #         entity_positive_list += [sen]
            #     elif sen == 'NEGATIVE':
            #         entity_negative_count  += [sen]
            #     else:
            #         entity_neutral_count  += [sen]

            #     st.write(ent)
            #     st.write(sen)
            #     entity_sentiment_claims = support_claims(article['Content'], ent, sen)
            #     st.write(f"<div style='height: 80px;font-size: 30px;text-align: left; padding:20px'>{ent} ({sen})</div>", unsafe_allow_html=True) 
            #     st.write(entity_sentiment_claims)


     
            # summary_col1, summary_col2 = st.columns([6,3])
    
            
            # focused_summary_toggle = summary_col1.toggle('Make focused summary', value=False, key = 'foccused_summary_key')            
            # summary_button = summary_col2.button('Summarize article',use_container_width = True, key = 'summary_button_key')
            
            # focus = None
            # if focused_summary_toggle:
            #     focus = summary_col1.text_input('Input summary focus', value='', key = 'focus_key')                
            #     if focus == '':
            #         focus = None            
            # if summary_button:
            #     s = GPTSummarizer(
            #         model='gpt-3.5-turbo', max_words=50, focus=focus
            #     )

            #     summary_col1.write('Summary')
            #     article_summary = s.fit_transform([article['Content']])[0]
            #     st.write(article_summary)
            
            # st.write('Article content')
            st.caption('**Content**')  
            st.write(article['Content'])


    else:
        st.write('No articles found')
    


def ml_news_classifier():
    st.title('Summarizing 2018 Articles')
    session_state = get_session_state()
    
    sf_col1, sf_col2, sf_col3 = st.columns([2,2,5])
    session_state = get_session_state()
    filter_headlines = sf_col1.toggle('Filter Articles', value=session_state.filter_article_summarization, key = 'filter_article_summarization_key')
    df_filtered = df 

    if filter_headlines == True:
        show_common_filter= sf_col2.toggle('Show Filter', value=False, key = 'show_common_filter_summarization_key')    
        
        if show_common_filter == True:
             common_filter()        
        df_filtered = session_state.df_filtered

    headline_temp = None
    if len(df_filtered) >0:
        headlines = df_filtered['Headline'].to_list()
        art_col1, art_col2 = st.columns([7,2])
        
        headline_temp = art_col1.selectbox('Select article title', headlines , index=session_state.headline_index)
        
        art_col2.write("<div style='height:30px'> </div>", unsafe_allow_html=True)           
        apply_article = art_col2.button('Apply',use_container_width = True, key = 'article_apply_key')
       
        if apply_article:
            session_state.headline = headline_temp            
            if headline_temp is None:
                session_state.headline_index = None
                session_state.article = None
            else:
                session_state.headline_index = headlines.index(headline_temp)
                article_temp = df_filtered[df_filtered['Headline']==headline_temp].iloc[0]
                session_state.article = article_temp       

            
        if apply_article and session_state.article is not None:
            article = session_state.article 
            st.header(f"[{article['Headline']}]({article['URL']})")
            st.caption(f"__Published date:__ {article['Date']}")


def interactive_quiz():
    st.title('Interactive quiz: Test your knowledge')
    st.title('Summarizing 2018 Articles')
    session_state = get_session_state()
    
    sf_col1, sf_col2, sf_col3 = st.columns([2,2,5])
    session_state = get_session_state()
    filter_headlines = sf_col1.toggle('Filter Articles', value=session_state.filter_article_summarization, key = 'filter_article_summarization_key')
    df_filtered = df 

    if filter_headlines == True:
        show_common_filter= sf_col2.toggle('Show Filter', value=False, key = 'show_common_filter_summarization_key')    
        
        if show_common_filter == True:
             common_filter()        
        df_filtered = session_state.df_filtered

    headline_temp = None
    if len(df_filtered) >0:
        headlines = df_filtered['Headline'].to_list()
        art_col1, art_col2 = st.columns([7,2])
        
        headline_temp = art_col1.selectbox('Select article title', headlines , index=session_state.headline_index)
        
        art_col2.write("<div style='height:30px'> </div>", unsafe_allow_html=True)           
        apply_article = art_col2.button('Apply',use_container_width = True, key = 'article_apply_key')
       
        if apply_article:
            session_state.headline = headline_temp            
            if headline_temp is None:
                session_state.headline_index = None
                session_state.article = None
            else:
                session_state.headline_index = headlines.index(headline_temp)
                article_temp = df_filtered[df_filtered['Headline']==headline_temp].iloc[0]
                session_state.article = article_temp       

            
        if apply_article and session_state.article is not None:
            article = session_state.article 
            st.header(f"[{article['Headline']}]({article['URL']})")
            st.caption(f"__Published date:__ {article['Date']}")# Main function to run the Streamlit app
def main():

    my_page = st.sidebar.radio('Page Navigation',
                           ['About the data', 'Interactive highlights', 
                            'News summarization', 
                            'ML News Classifier',
                            'Interactive quiz: Test your knowledge'])



    if my_page == "About the data":
        about_the_data()
    elif my_page == "Interactive highlights":
        interactive_highlights()
    elif my_page == "News summarization":
        news_summarization()
    elif my_page == "ML News Classifier":
        ml_news_classifier()
    elif my_page == "Interactive quiz: Test your knowledge":
        interactive_quiz()
        
if __name__ == "__main__":
    main()




