import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def treinar_modelo():
    data = {
        'renda': [5000, 3000, 7000, 4000, 6000, 12000, 8000, 3500, 5000, 4500],
        'idade': [25, 30, 45, 35, 50, 60, 40, 28, 32, 29],
        'historico_credito': [0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
        'divida_atual': [1000, 2000, 500, 1500, 1000, 2500, 1500, 800, 1200, 700],
        'inadimplente': [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    }

    df = pd.DataFrame(data)

    X = df[['renda', 'idade', 'historico_credito', 'divida_atual']]
    y = df['inadimplente']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    print(report)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_matrix)

    messagebox.showinfo("Treinamento Concluído", "O modelo foi treinado com sucesso!")

    return model, scaler


def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Adimplente", "Inadimplente"], yticklabels=["Adimplente", "Inadimplente"])
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.show()


def prever_risco():
    try:
        renda = float(entry_renda.get())
        idade = int(entry_idade.get())
        historico_credito = int(entry_historico_credito.get())
        divida_atual = float(entry_divida_atual.get())

        if 'modelo' not in globals():
            messagebox.showwarning("Erro", "O modelo precisa ser treinado primeiro.")
            return

        dados_cliente = np.array([[renda, idade, historico_credito, divida_atual]])
        dados_cliente = scaler.transform(dados_cliente)

        previsao = modelo.predict(dados_cliente)

        if previsao[0] == 1:
            messagebox.showinfo("Resultado", "Este cliente tem alto risco de inadimplência!")
        else:
            messagebox.showinfo("Resultado", "Este cliente tem baixo risco de inadimplência.")

    except ValueError:
        messagebox.showwarning("Erro", "Por favor, preencha todos os campos corretamente!")


def plot_training_data():
    data = {
        'renda': [5000, 3000, 7000, 4000, 6000, 12000, 8000, 3500, 5000, 4500],
        'idade': [25, 30, 45, 35, 50, 60, 40, 28, 32, 29],
        'historico_credito': [0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
        'divida_atual': [1000, 2000, 500, 1500, 1000, 2500, 1500, 800, 1200, 700],
        'inadimplente': [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    }

    df = pd.DataFrame(data)

    sns.pairplot(df, hue='inadimplente')
    plt.show()


root = tk.Tk()
root.title("Análise de Risco de Inadimplência")
root.geometry("400x350")

label_titulo = tk.Label(root, text="Análise de Risco de Inadimplência", font=("Arial", 16))
label_titulo.pack(pady=10)

label_renda = tk.Label(root, text="Renda Mensal (R$):")
label_renda.pack(pady=5)
entry_renda = tk.Entry(root)
entry_renda.pack(pady=5)

label_idade = tk.Label(root, text="Idade:")
label_idade.pack(pady=5)
entry_idade = tk.Entry(root)
entry_idade.pack(pady=5)

label_historico_credito = tk.Label(root, text="Histórico de Crédito (0: bom, 1: ruim):")
label_historico_credito.pack(pady=5)
entry_historico_credito = tk.Entry(root)
entry_historico_credito.pack(pady=5)

label_divida_atual = tk.Label(root, text="Dívida Atual (R$):")
label_divida_atual.pack(pady=5)
entry_divida_atual = tk.Entry(root)
entry_divida_atual.pack(pady=5)

botao_treinar = tk.Button(root, text="Treinar Modelo", command=lambda: treinar_modelo())
botao_treinar.pack(pady=10)

botao_prever = tk.Button(root, text="Prever Risco", command=lambda: prever_risco())
botao_prever.pack(pady=10)

botao_plot = tk.Button(root, text="Visualizar Dados de Treinamento", command=lambda: plot_training_data())
botao_plot.pack(pady=10)

modelo = None
scaler = None

root.mainloop()
