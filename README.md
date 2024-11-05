<h1>Documentação do Programa</h1>
- Este programa implementa e compara dois modelos de aprendizado de máquina (SVM e Regressão Logística) usando o conjunto de dados Iris. A seguir está uma descrição detalhada do código e suas funcionalidades.

- Classes e Métodos
Classe Modelo
A classe Modelo encapsula o processo de carregamento de dados, pré-processamento, treinamento e teste de modelos de aprendizado de máquina.

__init__(self, tipo_modelo='svm')
Inicializa a instância do modelo.

tipo_modelo (str): Tipo de modelo a ser usado ('svm' ou 'linear').
CarregarDataset(self, path)
Carrega o conjunto de dados a partir de um arquivo CSV.

path (str): Caminho do arquivo CSV.
- TratamentoDeDados(self)
Realiza o pré-processamento dos dados, incluindo:

- Remoção de valores nulos.
Codificação da coluna Species para valores numéricos.
- Separação das features (X) e do target (y).
Treinamento(self)
Divide os dados em conjuntos de treino e teste, e treina o modelo apropriado com base em tipo_modelo.

Teste(self)
Faz predições no conjunto de teste e calcula a acurácia do modelo.

Train(self)
Fluxo principal para o treinamento do modelo, que inclui:

Carregar o dataset.
Realizar o tratamento de dados.
Treinar o modelo.
Função Principal
comparar_modelos()
Função que cria instâncias da classe Modelo para SVM e Regressão Logística, treina e testa cada modelo, e imprime os resultados.

- Exemplo de Uso
Para usar o programa, execute a função comparar_modelos(), que compara os resultados de ambos os modelos.

comparar_modelos()
Este script será executado automaticamente quando o arquivo main.py for executado, graças à chamada da função comparar_modelos() no final do script.

Dependências
pandas
scikit-learn
Certifique-se de instalar esses pacotes antes de executar o programa:

pip install pandas scikit-learn
Esta documentação fornece uma visão geral do funcionamento do código e como utilizá-lo. Para mais detalhes, consulte o código-fonte.
