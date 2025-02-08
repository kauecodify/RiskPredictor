# Análise de Risco de Inadimplência

Este projeto usa aprendizado de máquina para prever o risco de inadimplência de um cliente com base em dados financeiros como renda, idade, histórico de crédito e dívida atual. A aplicação é feita utilizando a biblioteca `Tkinter` para a interface gráfica, o `RandomForestClassifier` para a modelagem e `seaborn` para a visualização de dados e a matriz de confusão.

## Funcionalidades


![Captura de tela 2025-02-07 224420](https://github.com/user-attachments/assets/11017f45-ef75-420e-8c11-87b56c099162)


- **Treinamento do Modelo**: O modelo de Random Forest é treinado usando dados de clientes fictícios, e o treinamento pode ser feito a partir da interface gráfica.
- **Previsão de Risco**: Após o treinamento, o usuário pode inserir os dados de um cliente e o modelo fará uma previsão sobre o risco de inadimplência (alto ou baixo).
- **Visualização de Dados de Treinamento**: O usuário pode visualizar os dados de treinamento em um gráfico de dispersão para analisar a relação entre as variáveis.
- **Matriz de Confusão**: A matriz de confusão é gerada automaticamente após o treinamento do modelo, fornecendo uma visão clara do desempenho do modelo.

## Requisitos

Para rodar o projeto, é necessário ter as seguintes bibliotecas instaladas:

- `tkinter`: para a interface gráfica (geralmente já instalada com o Python).
- `pandas`: para manipulação de dados.
- `numpy`: para manipulação de arrays numéricos.
- `scikit-learn`: para modelagem de aprendizado de máquina.
- `matplotlib` e `seaborn`: para visualização de dados.

## Dependências

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

MIT LICENSE ©


