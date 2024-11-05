import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

class Modelo():
    def __init__(self, tipo_modelo='svm'):
        """
        Inicializa o modelo.
        
        Parâmetros:
        - tipo_modelo (str): Tipo de modelo a ser usado ('svm' ou 'linear')
        """
        self.tipo_modelo = tipo_modelo

    def CarregarDataset(self, path):
        """
        Carrega o conjunto de dados a partir de um arquivo CSV.
        """
        try:
            # Carrega o dataset com o cabeçalho
            self.df = pd.read_csv(path)
            
            # Remove a coluna Id
            if 'Id' in self.df.columns:
                self.df = self.df.drop('Id', axis=1)
                
            # Remove a linha de cabeçalho se ela foi duplicada nos dados
            if self.df.iloc[0]['SepalLengthCm'] == 'SepalLengthCm':
                self.df = self.df.iloc[1:]
                
            # Converte as colunas numéricas para float
            numeric_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
            self.df[numeric_columns] = self.df[numeric_columns].astype(float)
            
            print(f"Dataset carregado com sucesso. Shape: {self.df.shape}")
            print("Primeiras linhas do dataset:")
            print(self.df.head())
        except Exception as e:
            print(f"Erro ao carregar dataset: {str(e)}")
            raise

    def TratamentoDeDados(self):
        """
        Realiza o pré-processamento dos dados.
        """
        try:
            # Verifica valores nulos
            if self.df.isnull().sum().any():
                self.df = self.df.dropna()
            
            # Converte a coluna Species para valores numéricos
            le = LabelEncoder()
            self.df['Species'] = le.fit_transform(self.df['Species'])
            
            # Separa features (X) e target (y)
            feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
            self.X = self.df[feature_columns].values
            self.y = self.df['Species'].values
            
            print("Tratamento de dados concluído com sucesso")
            print("Shape dos dados de treino:", self.X.shape)
        except Exception as e:
            print(f"Erro no tratamento de dados: {str(e)}")
            raise

    def Treinamento(self):
        # Divide os dados em treino e teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Seleciona e treina o modelo apropriado
        if self.tipo_modelo == 'svm':
            self.modelo = SVC(kernel='rbf', random_state=42)
        elif self.tipo_modelo == 'linear':
            # Mudando para LogisticRegression que é mais apropriado para classificação
            from sklearn.linear_model import LogisticRegression
            self.modelo = LogisticRegression(random_state=42)
        else:
            raise ValueError("Tipo de modelo não suportado")
            
        self.modelo.fit(self.X_train, self.y_train)

    def Teste(self):
        try:
            # Faz predições no conjunto de teste
            y_pred = self.modelo.predict(self.X_test)
            
            # Calcula e retorna a acurácia
            from sklearn.metrics import accuracy_score
            acuracia = accuracy_score(self.y_test, y_pred)
            print(f'Acurácia do modelo: {acuracia:.2f}')
        except Exception as e:
            print(f"Erro durante o teste: {str(e)}")

    def Train(self):
        """
        Função principal para o fluxo de treinamento do modelo.
        """
        try:
            print(f"\nIniciando treinamento do modelo {self.tipo_modelo}")
            self.CarregarDataset("iris.data")  # Carrega o dataset
            self.TratamentoDeDados()           # Importante: precisa ser chamado antes do Treinamento
            self.Treinamento()                 # Executa o treinamento
        except Exception as e:
            print(f"Erro durante o processo de treinamento: {str(e)}")

# Exemplo de uso:
def comparar_modelos():
    # Modelo SVM
    modelo_svm = Modelo(tipo_modelo='svm')
    modelo_svm.Train()
    print("Resultados do SVM:")
    modelo_svm.Teste()

    # Modelo de Regressão Linear
    modelo_linear = Modelo(tipo_modelo='linear')
    modelo_linear.Train()
    print("\nResultados da Regressão Linear:")
    modelo_linear.Teste()

comparar_modelos()
