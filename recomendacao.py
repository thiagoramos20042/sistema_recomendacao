import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# Carregando o dataset
df = pd.read_csv('Strength Training.csv')

# Convertendo as colunas para tipo numérico
df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')
df['no_of_ratings'] = pd.to_numeric(df['no_of_ratings'], errors='coerce')

# Preenchendo os valores ausentes com a média
df['ratings'].fillna(df['ratings'].mean(), inplace=True)
df['no_of_ratings'].fillna(df['no_of_ratings'].mean(), inplace=True)

# Selecionando os recursos para treinamento
features = ['ratings', 'no_of_ratings']

# Normalizando os recursos
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(df[features])
data_normalized_df = pd.DataFrame(data_normalized, columns=features)

# Treinando o modelo k-NN
knn_model = NearestNeighbors(metric='euclidean', algorithm='brute')
knn_model.fit(data_normalized_df)

# Função para recomendar produtos com base em nomes de produtos
def recommend_products_with_images(product_names, n_recommendations=3):
    all_recommendations = []
    for product_name in product_names:
        product_index = df[df['name'] == product_name].index[0]
        distances, indices = knn_model.kneighbors(data_normalized_df.iloc[product_index, :].values.reshape(1, -1), n_neighbors=n_recommendations+1)
        recommended_products = [{"name": df.iloc[i]['name'], "image_url": df.iloc[i]['image']} for i in indices.flatten()][1:] # excluding the input product
        all_recommendations.extend(recommended_products)
    return all_recommendations

# Função para a página de recomendação
def recommendation_page():
    st.title('Recomendador de Produtos')
    selected_products = st.multiselect('Selecione os produtos:', options=df['name'].tolist())
    if st.button('Obter Recomendações'):
        if selected_products:
            recommendations = recommend_products_with_images(selected_products)
            for rec in recommendations:
                st.write(f"Nome: {rec['name']}")
                st.image(rec['image_url'])
        else:
            st.warning('Por favor, selecione pelo menos um produto.')

# Função para a página de explicação
def explanation_page():
    st.title('Como Funciona o Sistema de Recomendação')
    st.write("""
    Este sistema de recomendação utiliza o algoritmo k-NN (k vizinhos mais próximos) para fornecer recomendações personalizadas. Abaixo está a explicação de como funciona todo o processo.
    """)

    st.header('Algoritmo k-NN (k Vizinhos Mais Próximos)')
    st.write("""
    O algoritmo k-NN é um método de aprendizado supervisionado utilizado para classificação e regressão. É conhecido por sua simplicidade e eficácia, especialmente em tarefas de recomendação.
    """)

    st.subheader('Como Funciona?')
    st.write("""
    - Definir o Valor de k: Escolhe-se o número de vizinhos a serem considerados (k).
    - Calcular Distâncias: As distâncias entre o ponto de consulta e todos os outros pontos no conjunto de dados são calculadas usando a distância euclidiana.
    - Selecionar Vizinhos: Os k pontos mais próximos (ou vizinhos) são selecionados.
    - Tomar Decisão: A classe majoritária ou a média dos valores dos k vizinhos é usada.
    """)

    st.latex(r"\text{{Distância}} = \sqrt{{(x_2 - x_1)^2 + (y_2 - y_1)^2}}")

    st.subheader('Aplicação no Sistema de Recomendação')
    st.write("""
    - Pré-processamento: Os recursos são normalizados.
    - Treinamento do Modelo: Utilizamos o NearestNeighbors do Scikit-learn.
    - Recomendação: O modelo encontra os k vizinhos mais próximos para cada produto selecionado.
    - Exibição: Os produtos recomendados são exibidos na interface do usuário.
    """)

    st.header('Aplicações Práticas do Algoritmo k-NN')
    st.write("""
    - Recomendação de Produtos em Lojas Online: Recomenda produtos semelhantes aos clientes.
    - Diagnóstico Médico para Detecção de Doenças: Auxilia no diagnóstico preciso.
    - Reconhecimento Facial em Sistemas de Segurança: Reforça a segurança.
    - Avaliação de Crédito em Bancos: Ajuda na tomada de decisões sobre concessão de crédito.
    - Otimização de Consumo de Energia: Promove a eficiência energética.
    """)

    st.header('Conclusão')
    st.write("""
    O algoritmo k-NN é uma ferramenta poderosa e versátil que pode ser facilmente aplicada para tarefas de recomendação. Sua simplicidade e a capacidade de trabalhar bem com diferentes tipos de dados o tornam uma escolha popular em muitas aplicações de aprendizado de máquina, desde o comércio e a saúde até a segurança e a gestão de energia.
    """)
# Função para a página "Problema de Negócio"
def business_problem_page():
    st.title("Problema de Negócio Hipotético")
    
    st.subheader("Contexto")
    st.write("""
    A "FitnessGear Inc.", uma empresa líder em equipamentos de treinamento de força, enfrentava um problema significativo com sua plataforma de comércio eletrônico. Embora a empresa oferecesse uma ampla variedade de produtos, desde pesos livres até máquinas sofisticadas, eles estavam enfrentando baixas taxas de conversão e uma média de venda por cliente abaixo das expectativas.

    A análise dos padrões de navegação revelou que os clientes frequentemente se sentiam sobrecarregados com a vasta seleção de produtos e tinham dificuldade em encontrar itens que se adequassem às suas necessidades específicas. Como resultado, muitos visitantes saíam do site sem fazer uma compra, ou compravam apenas os itens mais populares, perdendo produtos complementares que poderiam ser de seu interesse.
    """)

    st.subheader("Solução Implementada")
    st.write("""
    A empresa decidiu implementar um sistema de recomendação personalizado utilizando o algoritmo k-NN (k vizinhos mais próximos). Esta abordagem permitiu que a "FitnessGear Inc." analisasse as preferências individuais dos clientes, tais como avaliações e histórico de compras, e oferecesse recomendações de produtos personalizadas.

    O sistema recomendava produtos semelhantes, complementares e muitas vezes ignorados que estavam alinhados com as preferências e necessidades individuais do cliente.
    """)

    st.subheader("Retorno Financeiro")
    st.write("""
    A implementação do sistema de recomendação teve um impacto profundo nos resultados financeiros da empresa:

    - **Aumento nas Vendas Cruzadas**: Ao sugerir produtos complementares, como acessórios ou itens relacionados, o valor médio do carrinho de compras aumentou em 25%.
    - **Melhoria na Taxa de Conversão**: A taxa de conversão do site aumentou em 18%, já que os clientes encontraram mais facilmente produtos que atendiam às suas necessidades.
    - **Crescimento na Retenção de Clientes**: A experiência de compra personalizada melhorou a satisfação do cliente, resultando em um aumento de 10% na retenção de clientes.
    - **Expansão do Catálogo Vendido**: Produtos anteriormente ignorados ou menos populares começaram a ser vendidos, aumentando a diversidade de vendas e reduzindo o excesso de estoque em 15%.
    - **Aumento no Lucro Total**: Combinando todos esses fatores, a "FitnessGear Inc." experimentou um aumento de 30% no lucro líquido no primeiro ano após a implementação do sistema de recomendação.
    """)

def infos():
    st.title("Quem Sou Eu")
    
    st.subheader("Thiago Ramos de Oliveira")
    st.write("""
    🔬 Meu objetivo é fornecer soluções eficientes e escaláveis que impulsionem a tomada de decisões baseada em dados.

💻 Como programador em Python, aproveito seu poder e flexibilidade para desenvolver soluções de ponta a ponta. Minha experiência inclui a criação de scripts e pipelines de extração, transformação e carga (ETL), bem como o desenvolvimento de APIs para facilitar o acesso aos dados.

📊 A análise de dados é minha paixão. Utilizando técnicas estatísticas avançadas e ferramentas de visualização, tenho a capacidade de identificar padrões, tendências e insights ocultos nos dados. Ao traduzir informações complexas em relatórios e apresentações claras, ajudo as equipes a tomar decisões informadas e estratégicas.

🤖 A criação de modelos de machine learning é outra área em que me destaco. Utilizando algoritmos de aprendizado supervisionado e não supervisionado, desenvolvo modelos preditivos e de segmentação que melhoram a compreensão do negócio e impulsionam a eficiência operacional.

🚀 Sou apaixonado por aprender e estar atualizado com as mais recentes tendências e tecnologias relacionadas à analise de dados, machine learning, inteligência artificial. Sempre estou em busca de novos desafios que me permitam aplicar minha expertise e colaborar com equipes multidisciplinares em projetos inovadores.


Perfil no Linkedin:
https://www.linkedin.com/in/thiago-ramos-oliveira/

Portfólio no github:
https://github.com/thiagoramos20042?tab=repositories
    """)


# Páginas
st.sidebar.title('Menu')
page = st.sidebar.selectbox('Escolha uma página:', ['Problema de Negócio', 'Como Funciona', 'Recomendação de Produtos', 'Quem Sou Eu'])
if page == 'Recomendação de Produtos':
    recommendation_page()
elif page == 'Como Funciona':
    explanation_page()
elif page == 'Problema de Negócio':
    business_problem_page()
elif page == 'Quem Sou Eu':
    infos()