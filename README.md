1. Importando as Bibliotecas Necessárias
python
Copiar código
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
2. Carregando o Dataset
python
Copiar código
# O dataset pode ser baixado do Kaggle e carregado localmente.
# df = pd.read_csv('Online_Retail.csv')

# Como exemplo, vamos usar um dataset de amostra:
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
df = pd.read_excel(url)

# Visualizar as primeiras linhas
df.head()
3. Limpeza e Preparação dos Dados
python
Copiar código
# Removendo entradas nulas
df.dropna(inplace=True)

# Filtrando apenas transações positivas
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# Criando uma coluna de receita
df['Revenue'] = df['Quantity'] * df['UnitPrice']

# Resumo dos dados
df.describe()
4. Análise RFM (Recency, Frequency, Monetary)
python
Copiar código
# Criando a Tabela RFM
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
    'InvoiceNo': 'count',
    'Revenue': 'sum'
}).reset_index()

# Renomeando colunas
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Visualizando as primeiras linhas da Tabela RFM
rfm.head()
5. Normalizando os Dados
python
Copiar código
# Normalizando os dados RFM
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Convertendo de volta para DataFrame
rfm_scaled = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])

# Visualizando as primeiras linhas dos dados normalizados
rfm_scaled.head()
6. Aplicando K-means para Clusterização
python
Copiar código
# Definindo o modelo K-means
kmeans = KMeans(n_clusters=4, random_state=42)

# Ajustando o modelo
kmeans.fit(rfm_scaled)

# Atribuindo rótulos aos clusters
rfm['Cluster'] = kmeans.labels_

# Visualizando as primeiras linhas com os clusters
rfm.head()
7. Visualizando os Clusters
python
Copiar código
# Visualizando os clusters
sns.pairplot(rfm, hue='Cluster', palette='tab10')
plt.show()

# Tamanho de cada cluster
rfm['Cluster'].value_counts()
8. Conclusões e Próximos Passos
Após a análise, podemos observar como os clientes foram segmentados em diferentes clusters. Cada cluster representa um grupo de clientes com comportamentos de compra semelhantes, o que pode ser utilizado para estratégias de marketing direcionadas.

Cluster 0: Pode representar clientes de alto valor (alta frequência e alta receita).
Cluster 1: Clientes com alta frequência, mas baixo valor monetário.
Cluster 2: Clientes recentes com baixo valor.
Cluster 3: Clientes que não compram há muito tempo.
9. Salvando os Resultados
python
Copiar código
# Salvando os resultados em um arquivo CSV
rfm.to_csv('rfm_clusters.csv', index=False)
10. Considerações Finais
Este projeto demonstra como realizar uma segmentação de clientes utilizando análise RFM e K-means clustering. Essa abordagem pode ser aplicada a diversos tipos de dados de comportamento de compra para ajudar empresas a entender melhor seus clientes e criar estratégias de marketing mais eficazes.

Fontes
Online Retail Dataset - Kaggle
Documentação do K-means - Scikit-learn
