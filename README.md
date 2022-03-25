# Classificação de animais tilizando uma Rede Neural Convolucional


![image](https://img.shields.io/github/languages/top/stopasola/minimum-cost-graph)

## Introdução

   Este trabalho tem o objetivo de propor um algoritmo que seja capaz de
distinguir tipos diferentes de animais a partir de imagens disponíveis em um banco
de dados. O algoritmo deve, a partir de um certo número de amostras usadas como
treinamento, ser capaz de afirmar qual o tipo de animal presente em uma imagem,
por exemplo, se o animal presente na imagem é um cachorro ou um gato.
Para a resolução da proposta escolhida, foi usada uma rede neural
convolucional [1], tal especialização de redes neurais é amplamente utilizada na área
da visão computacional. A diferença primordial dessa rede neural comparada a
outras redes disponíveis é que ela consiste no pré-processamento das imagens de
entrada, que sofrem modificações por meio de filtros matriciais aplicados, que
acentuam características relevantes para o domínio definido. Após o processamento
inicial, a matriz da imagem é gerada com suas características.

## Desenvolvimento
   A arquitetura da rede neural convolucional (CNN - Convolutional Neural
Network) é como uma rede neural densa, porém, se diferencia no que tange ao
tratamento das imagens que serão reconhecidas. Considere a seguinte imagem:

<p align="center">
  <img width="250" height="250" src="https://user-images.githubusercontent.com/17886190/160052012-437783c1-a1b3-4902-ad50-cf774d8d907e.png">
</p>

   Uma rede neural densa utilizaria todas as posições da imagem como entrada,
tal imagem possui 32 bits de largura e 32 bits de altura e os 3 canais RGB, assim
teríamos 32 x 32 = 1024 x 3 = 3072 entradas para uma rede neural convencional,
considerando uma base de dados grande o processamento se torna altamente
custoso.
Dessa forma utilizamos a rede neural convolucional, que possui as seguintes
características:

● Não usa todas as entradas, ou seja, não usa todos os pixels da imagem

● Usa uma rede neural tradicional, mas no começo do processamento
transforma os dados na camada de entrada

● Vai descobrir as características mais importantes da imagem
A rede neural convolucional possui 4 etapas, elas são o operador de
convolução, Pooling, Flattening e a Rede Neural Densa.
 
   O Operador de Convolução realiza o processo de adicionar cada imagem para
seus vizinhos, ponderado por um detector de características (kernel). A imagem é
uma matriz e o detector de características é outra matriz

<p align="center">
  <img width="450" height="200" src="https://user-images.githubusercontent.com/17886190/160052073-849782de-19f5-4420-bb82-3615b7bf5084.png">
</p>


   A imagem resultante da operação de convolução é o mapa de características
(filter map), que é uma imagem menor que a original para facilitar o processamento.
Pode-se perder informações sobre a imagem, porém, o propósito é detectar as partes
principais, assim, o filtro preserva as características principais.
Após isso, aplica-se a operação Relu no mapa de características, que consiste
em transformar os valores negativos em 0 e manter os valores 0 e positivos da
imagem. A rede neural não utiliza apenas 1 detector de características mas sim vários
e seleciona o melhor.

   O Pooling serve para enfatizar ainda mais as características dos objetos ou das 
imagens que se quer classificar, assim, ele seleciona as características mais relevantes
e reduz o overfitting e ruídos desnecessários. Ele é aplicado na imagem resultante da
operação Relu e gera uma imagem menor. Utiliza-se o Max Pooling pois ele seleciona
os maiores valores da matriz (imagem), focando nas características mais relevantes,
porém, existem outras opções como o mínimo e o media.
   O Flattening transforma a imagem resultante do Pooling em um vetor onde
cada posição servirá como entrada para a rede neural densa.

###### Estrutura da CNN utilizada
   Na primeira camada da rede nós temos o operador de convolução, e foram
utilizados 32 detectores de características, isso gera 32 mapas de características e o
melhor é escolhido. O tamanho do mapa de características escolhido é uma matriz
2x2.
   Na etapa de Pooling, a matriz utilizada é de tamanho 2x2 para selecionar as
características mais relevantes.
Por questões de otimização aplicamos novamente uma camada de convolução
nos mesmos moldes da primeira, com 32 mapas de características e em seguida
novamente é aplicado o Pooling com uma matriz 2x2, após isso, é realizado o
Flattening.
   Na rede neural, a primeira camada possui 128 neurônios, há ainda uma
camada oculta nos mesmos moldes da primeira com 128 neurônios, a camada de
saída possui 3 neurônios, pois temos 3 classes de saída.

###### Codificação dos dados de entrada

   Foi utilizada a base de dados com imagens de cães, gatos e pandas [2], onde
cada classe possui 1000 imagens, e a codificação é feita reduzindo as imagens
originais para uma dimensão de 32x55 pixels, para que essas imagens sejam tratadas
pela rede neural convolucional.

###### Codificação das saídas da rede
   A CNN possui uma camada de saída com 3 neurônios onde um se refere a
classe cães, outro a classe gatos e outro a classe pandas, assim, a imagem na entrada
é classificada com uma dessas classes.

###### Algoritmo de treinamento utilizado
   O algoritmo de treinamento utilizado é a rede neural feedforward densa, que
é uma rede onde todos os neurônios de uma camada são conectados com todas as
entradas de uma camada.

<p align="center">
  <img width="350" height="250" src="https://user-images.githubusercontent.com/17886190/160052211-af57e786-3384-4111-bbc6-c154c1148ebb.png">
</p>

###### Função de ativação
   Para a camada de entrada e a camada intermediária foram utilizadas a função
ReLU que consiste em transformar todos os valores negativos em 0 e manter os
positivos e 0, enquanto a camada de saída utiliza a função softmax, utilizada em
redes neurais de classificação, ela faz com que a rede neural calcule a probabilidade
dos dados serem de uma das classes definidas.

###### Taxa de aprendizado
   A taxa de aprendizado utilizada é a taxa padrão da função fit da biblioteca
keras, que é 0.01. Além disso, o parâmetro época foi definido como 25.


## Resultado da implementação

   Foram realizados 10 testes com a rede neural convolucional sobre a base de
dados dos animais. Os testes foram executados em uma máquina com as seguintes
especificações:

Processador: Intel(R) Core(TM) i3-7100U CPU 2.40GHz
Memória Ram: 4,00 GB DDR3 1600 MHz
Sistema operacional: Windows 10

E os resultados dos testes executados foram:

1º teste: 71.50%

<p align="center">
  <img width="400" height="400" src="https://user-images.githubusercontent.com/17886190/160052325-9b8678e4-f905-4316-a4c2-8ecf92d810f5.png">
</p>

2º teste: 73.67%


<p align="center">
  <img width="400" height="400" src="https://user-images.githubusercontent.com/17886190/160052354-4ea42795-a8ef-446a-94a8-31bf57a731c7.png">
</p>


3º teste: 75.00%

<p align="center">
  <img width="400" height="400" src="https://user-images.githubusercontent.com/17886190/160052372-f7f7b81a-1f34-47ee-ae82-89d014f16be7.png">
</p>

   Em 10 testes realizados a média de acurácia da rede neural convolucional é de
73,218%, sendo a melhor acurácia de 75% e a menor de 70,17%.

## Conclusão

   A rede neural apresenta uma acurácia acima de 70% em todos os casos
testados, havendo uma variação para cada teste devido ao modo aleatório de se
dividir as imagens de treino e de teste. Devido a ser uma base de dados pequena
comparada com outras, a possibilidade de variação das imagens no treinamento é
menor, havendo mais chances para erros. Assim, com uma base de dados maior os
resultados poderiam atingir melhor acurácia.


## Referências bibliográficas

[1] A Comprehensive Guide to Convolutional Neural Networks. Disponível em
[<https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-n
etworks-the-eli5-way-3bd2b1164a53>](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-n
etworks-the-eli5-way-3bd2b1164a53). Acesso em: 27/08/2021.

[2] Animal Image Dataset (DOG, CAT and Panda). Kaggle, 2021. Disponível em:
<[https://www.kaggle.com/ashishsaxena2209/animal-image-datasetdog-cat-and-panda](https://www.kaggle.com/ashishsaxena2209/animal-image-datasetdog-cat-and-panda)>. Acesso em: 25/08/2021.

[3] Tensor Flow. Tensor Flow, 2021. Disponível em:
<[https://www.tensorflow.org/?hl=pt-br](https://www.tensorflow.org/?hl=pt-br)> . Acesso em: 25/08/2021

[4] Keras. Keras, 2021. Disponível em: <[https://keras.io](https://keras.io)>. Acesso em: 24/08/2021

[5] Python. Python Software Foundation. Disponível em: <[https://www.python.org](https://www.python.org)>
