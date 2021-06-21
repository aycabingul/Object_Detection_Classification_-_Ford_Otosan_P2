# Project
## -   What is Machine Learning?
  With machine learning, humans input data as well as the answers expected from the data, and out come the rules. These rules can then be applied to new data to produce original answers. A machine-learning system is trained rather than explicitly programmed. It’s presented with many examples relevant to a task, and it finds statistical structure in these examples that eventually allows the system to come up with rules for automating the task. 

(Makine Öğrenmesi, veri ve beklenen çıktıları kullanarak yeni verilere özgün cevaplar üretecek kuralları öğrenir. Makine öğrenmesi sistemleri elle programlanmak yerine **eğitilirler**. Göreve yönelik verilen örneklere bakarak istatistiksel örüntüleri ortaya çıkartıp görevi otomatikleştirecek kuralları ortaya koyar.) 
## -   What is Unsupervised vs Supervised learning difference?
### Supervised learning;
It consists of learning to map input data to known targets (also called annotations), given a set of examples (often annotated by humans).


(Bir veri seti üzerindeki girdilerden bilinen çıktılara/hedeflere olan eşleştirmeyi öğrenmeyi kapsar.

### Unsupervised learning;
This branch of machine learning consists of finding interesting transformations of the input data without the help of any targets, for the purposes of data visualization, data compression, or data denoising, or to better understand the correlations present in the data at hand.
The main distinction between the two approaches is the use of labeled datasets. To put it simply, supervised learning uses labeled input and output data, while an unsupervised learning algorithm does not.


( Makine öğrenmesinin bu kategorisinin amacı veriyi görselleştirmek, verileri sıkıştırmak, verilerdeki gürültüyü azaltmak ya da eldeki veriler arasındaki kolerasyonu anlamak için girdilerden herhangi bir bilinen hedefin yardımı olmaksızın amaca uygun dönüşümleri bulmaktır.)
(İki yaklaşım arasındaki temel ayrım, etiketli veri kümelerinin kullanılmasıdır. Basitçe söylemek gerekirse, denetimli öğrenme etiketli girdi ve çıktı verilerini kullanırken denetimsiz öğrenme algoritması kullanmaz.)
## -   What is Deep Learning?
Deep learning is a specific subfield of machine learning: a new take on learning representations from data that puts an emphasis on learning successive layers of increasingly meaningful representations. The deep in **deep learning** isn’t a reference to any kind of deeper understanding achieved by the approach; rather, it stands for this idea of successive layers of representations. How many layers contribute to a model of the data is called the *depth* of the model.


(Derin öğrenme, birbirini takip eden katmanlarda veriler işlenirken giderek artan şekilde daha kullanışlı gösterimler elde edilebilen makine öğrenmesinin bir alt alanıdır. Derin derken kastedilen **derin öğrenmenin** birtakım derin bilgiler elde etmesi değil, birbirini takip eden gösterim katmanları ifade edilmektedir. Modeldeki katman modelin *derinliğini* oluşturmaktadır.)
## -   What is Neural Network (NN)?
Modern deeplearning often involves tens or even hundreds of successive layers of representations and they’re all learned automatically from exposure to training data. Meanwhile, other approaches to machine learning tend to focus on learning only one or two layers of representations of the data; hence, they’re sometimes called shallow learning. In deep learning, these layered representations are (almost always) learned via models called neural networks, structured in literal layers stacked on top of each other.

Artificial neural networks (ANNs) are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network.


(Modern derin öğrenme modelleri, onlarca hatta yüzlerce birbirini takip eden katmanlar içermektedir. Oysa diğer makine öğrenme algoritmaları, genelde bir veya iki katmandan oluşur ki bazen *sığ öğrenme* olarak da adlandırılırlar. Bu katmanlı gösterim, **sinir ağı** diye adlandırılan ve birbirini takip eden katmanları olan model sayesinde öğrenilmektedir.

Yapay sinir ağları (YSA), bir girdi katmanı, bir veya daha fazla gizli katman ve bir çıktı katmanı içeren bir düğüm katmanından oluşur. Her düğüm veya yapay nöron diğerine bağlanır ve ilişkili bir ağırlık ve eşiğe sahiptir. Herhangi bir düğümün çıktısı belirtilen eşik değerinin üzerindeyse, o düğüm etkinleştirilir ve ağın bir sonraki katmanına veri gönderilir. Aksi takdirde, ağın bir sonraki katmanına hiçbir veri iletilmez.)
![enter image description here](https://miro.medium.com/max/1612/0*3rRcXavz8hmHXAFL.png)
## -   What is Convolution Neural Network (CNN)? 
A convolutional neural network (ConvNet / Convolutional neural networks -CNN) is a deep learning algorithm that can take an input image and separate various aspects/objects in the image.
Convolutional neural networks are deep neural networks that are mainly used to classify images (for example, to name what they see), cluster by similarity (photo search), and perform object recognition in scenes.

--   CNN learns the filters automatically without mentioning it explicitly. These filters help in extracting the right and relevant features from the input data.
--  CNN captures the  **spatial features**  from an image. Spatial features refer to the arrangement of pixels and the relationship between them in an image. They help us in identifying the object accurately, the location of an object, as well as its relation with other objects in an image.


(Bir evrişimsel sinir ağı (ConvNet / Convolutional neural networks -CNN), bir girdi görüntüsünü alıp, görüntüdeki çeşitli görünüşleri/nesneleri birbirinden ayırabilen derin öğrenme algoritmasıdır.
Evrişimli sinir ağları, temel olarak görüntüleri sınıflandırmak (örneğin gördüklerini isimlendirmek), benzerlikle kümelemek (fotoğraf arama) ve sahnelerde nesne tanıma yapmak için kullanılan derin yapay sinir ağlarıdır.

-- CNN, filtreleri açıkça belirtmeden otomatik olarak öğrenir. Bu filtreler, giriş verilerinden doğru ve ilgili özelliklerin çıkarılmasına yardımcı olur.
--  CNN bir görüntüden **uzamsal özellikleri** yakalar . Uzamsal özellikler, bir görüntüdeki piksellerin düzenini ve aralarındaki ilişkiyi ifade eder. Nesneyi, bir nesnenin konumunu ve bir görüntüdeki diğer nesnelerle ilişkisini doğru bir şekilde tanımlamamıza yardımcı olurlar.)
 ![enter image description here](https://www.researchgate.net/publication/326152216/figure/fig2/AS:644307941879809@1530626392505/Deep-convolutional-neural-network-DCNN-architecture-A-schematic-diagram-of-AlexNet.png)
 ## What is segmentation task in NN?
 Image segmentation is the process of classifying each pixel in an image belonging to a certain class and hence can be thought of as a classification problem per pixel. semantic segmentation falls under supervised learning. There are two types of segmentation techniques;
 1.  **Semantic segmentation**: Semantic segmentation is the process of classifying each pixel belonging to a particular label. It doesn't different across different instances of the same object. For example if there are 2 cats in an image, semantic segmentation gives same label to all the pixels of both cats
![enter image description here](https://wiki.tum.de/download/attachments/23561833/sms.png?version=1&modificationDate=1483619907233&api=v2)
 2. **Instance segmentation**: Instance segmentation differs from semantic segmentation in the sense that it gives a unique label to every instance of a particular object in the image. As can be seen in the image above all 3 dogs are assigned different colours i.e different labels. With semantic segmentation all of them would have been assigned the same colour.
 ![enter image description here](https://www.researchgate.net/profile/Vinorth-Varatharasan/publication/339328277/figure/fig1/AS:864554888204291@1583137356688/Semantic-segmentation-left-and-Instance-segmentation-right-8.ppm)

( Semantik Segmentasyon, belirli bir sınıfa ait bir görüntüdeki her bir pikseli sınıflandırma işlemidir ve bu nedenle piksel başına bir sınıflandırma problemi olarak düşünülebilir. Semantik segmentasyon denetimli öğrenme kapsamına girer. İki tür segmentasyon tekniği vardır;
1.  **Semantic segmentation**: Semantik segmentasyon, belirli bir etikete ait her pikseli sınıflandırma işlemidir. Aynı nesnenin farklı örneklerinde farklı değildir. Örneğin bir görüntüde 2 kedi varsa, anlamsal bölümleme her iki kedinin tüm piksellerine aynı etiketi verir.
2. **Instance segmentation**: Örnek bölümleme, görüntüdeki belirli bir nesnenin her örneğine benzersiz bir etiket vermesi anlamında anlamsal bölümlemeden farklıdır. Yukarıdaki resimde görüldüğü gibi 3 köpeğe de farklı renkler yani farklı etiketler atanmıştır. Anlamsal bölümleme ile hepsine aynı renk atanırdı.)

## What is classification task in NN? 
Artificial Neural Network applications are generally used in prediction, classification, data association, data interpretation and data filtering. Artificial neural networks used for classification take on the task of classifying input values. Classification is the categorization process in which objects are defined, differentiated and data is understood on the basis of the training set. Classification is a supervised learning technique with a training set and correctly defined observations.
All classification tasks depend upon labeled datasets; that is, humans must transfer their knowledge to the dataset in order for a neural network to learn the correlation between labels and data. This is known as supervised learning.
-   Detect faces, identify people in images, recognize facial expressions (angry, joyful)
-   Identify objects in images (stop signs, pedestrians, lane markers…)
-   Recognize gestures in video
-   Detect voices, identify speakers, transcribe speech to text, recognize sentiment in voices
-   Classify text as spam (in emails), or fraudulent (in insurance claims); recognize sentiment in text (customer feedback)


(Yapay Sinir Ağları uygulamaları genellikle tahmin, sınıflandırma, veri ilişkilendirme, veri yorumlama ve veri filtreleme işlemlerinde kullanılmaktadır.  Sınıflandırma amacıyla kullanılan yapay sinir ağları, girdi değerlerini sınıflama görevini üstlenirler. Sınıflandırma, nesnelerin tanımlandığı, farklılaştığı ve verilerin eğitim seti temelinde anlaşıldığı kategorizasyon sürecidir. Sınıflandırma, bir eğitim seti ve doğru tanımlanmış gözlemlerin bulunduğu denetimli bir öğrenme tekniğidir.
Tüm sınıflandırma görevleri, etiketlenmiş veri kümelerine bağlıdır; yani, bir sinir ağının etiketler ve veriler arasındaki ilişkiyi öğrenmesi için insanların bilgilerini veri kümesine aktarması gerekir. Bu, denetimli öğrenme olarak bilinir .
-   Yüzleri algılayın, görüntülerdeki insanları tanımlayın, yüz ifadelerini tanıyın (kızgın, neşeli)
-   Görüntülerdeki nesneleri tanımlayın (dur işaretleri, yayalar, şerit işaretleri…)
-   Videodaki hareketleri tanıma
-   Sesleri algılayın, konuşmacıları tanımlayın, konuşmayı metne dönüştürün, seslerdeki duyguyu tanıyın
-   Metni spam (e-postalarda) veya hileli (sigorta taleplerinde) olarak sınıflandırma; metindeki duyarlılığı tanıma (müşteri geri bildirimi).
![enter image description here](https://learnopencv.com/wp-content/uploads/2017/11/cnn-schema1.jpg)
## Compare segmentation and classification in NN.
The difference between segmentation and classification is clear at some extend. And there is a one difference between both of them. The classification process is easier than segmentation, in classification all objects in a single image is grouped or categorized into a single class. While in segmentation each object of a single class in an image is highlighted with different shades to make them recognizable to computer vision.


(Segmentasyon ve sınıflandırma arasındaki fark bir dereceye kadar açıktır. Ve ikisi arasında tek bir fark var. Sınıflandırma işlemi segmentasyondan daha kolaydır, sınıflandırmada tek bir görüntüdeki tüm nesneler tek bir sınıfa gruplanır veya kategorize edilir. Segmentasyon sırasında, bir görüntüdeki tek bir sınıfın her bir nesnesi, bilgisayarla görüye tanınabilmesi için farklı gölgelerle vurgulanır.)

## -   What is data and dataset difference?
-   **Data**  are observations or measurements (unprocessed or processed) represented as text, numbers, or multimedia.
- A  **dataset**  is a structured collection of data generally associated with a unique body of work.

## - What is the difference between supervised and unsupervised learning in terms of dataset?
**In a supervised learning** model, the algorithm learns on a labeled **dataset**, providing an answer key that the algorithm can use to evaluate its accuracy on training data. An **unsupervised** model, in contrast, provides unlabeled data that the algorithm tries to make sense of by extracting features and patterns on its own.


(Denetimli bir öğrenme modelinde, algoritma etiketli bir veri kümesi üzerinde öğrenir ve algoritmanın eğitim verileri üzerindeki doğruluğunu değerlendirmek için kullanabileceği bir cevap anahtarı sağlar. Denetimsiz bir model, aksine, algoritmanın kendi başına özellikleri ve kalıpları çıkararak anlamlandırmaya çalıştığı etiketlenmemiş verileri sağlar.)

# Data Preprocessing
## Extracting Masks
### -   What is color space ?
A **color space** is a coordinate system in which each color is represented as a single point. Colors are composed of a mixture of blue, green and red colors because they react differently at different wavelengths.
(**Renk uzayı**, her rengin tek bir nokta olarak temsil edildiği bir koordinat sistemidir. Renkler, farklı dalga boylarında farklı tepkime verdikleri için mavi, yeşil ve kırmızı renklerin karışımından oluşur.)
![enter image description here](https://www.hisour.com/wp-content/uploads/2018/03/RGB-color-space.jpg)
### -   What RGB stands for ?
**RGB** means Red Green Blue, ie the primary colors in additive color synthesis. A **RGB** file consists in composite layers of Red, Gree and Blue, each being coded on 256 levels from 0 to 255.
(**RGB**, Kırmızı Yeşil Mavi anlamına gelir, yani katkılı renk sentezindeki ana renkler. Bir **RGB** dosyası, her biri 0 ila 255 arasında 256 düzeyde kodlanmış Kırmızı, Yeşil ve Mavi bileşik katmanlarından oluşur.)

### -   In Python, can we transform from one color space to another?
There are more than 150 color-space conversion methods available in OpenCV. A popular computer vision library written in C/C++ with bindings for Python, OpenCV provides easy ways of manipulating color spaces.

(OpenCV'de 150'den fazla renk alanı dönüştürme yöntemi mevcuttur.
C/C++ ile Python için binding'lerle yazılmış popüler bir bilgisayarlı vision kütüphanesi olan OpenCV, renk uzaylarını değiştirmenin kolay yollarını sağlar.)

### -   What is the popular library for image processing?
OpenCV is one of the most famous and widely used open-source **libraries** for computer vision tasks such as **image processing**, object detection, face detection, **image** segmentation, face recognition, and many more. Other than this, it can also be used for machine learning tasks.


(OpenCV, görüntü işleme, nesne algılama, yüz algılama, görüntü bölütleme, yüz tanıma ve daha pek çok bilgisayarla görme görevleri için en ünlü ve yaygın olarak kullanılan açık kaynak kütüphanelerinden biridir. Bunun dışında makine öğrenimi görevleri için de kullanılabilir.)

#  Converting into Tensor
## -  What is Computational Graph.
A computational graph is a directed graph where the nodes correspond to **operations** or **variables**. Variables can feed their value into operations, and operations can feed their output into other operations. This way, every node in the graph defines a function of the variables.


(Bir hesaplama grafiği, düğümlerin işlemlere veya değişkenlere karşılık geldiği yönlendirilmiş bir grafiktir. Değişkenler değerlerini işlemlere besleyebilir ve işlemler çıktılarını diğer işlemlere besleyebilir. Bu şekilde, grafikteki her düğüm değişkenlerin bir fonksiyonunu tanımlar.)
![enter image description here](http://media5.datahacker.rs/2021/01/54-1.jpg)
## -   What is Tensor?
The values that are fed into the nodes and come out of the nodes are called **tensors**, which is just a fancy word for a multi-dimensional array. Hence, it subsumes scalars, vectors and matrices as well as tensors of a higher rank.


(Düğümlere beslenen ve düğümlerden çıkan değerlere **tensörler** denir, bu sadece çok boyutlu bir dizi için süslü bir kelimedir.)
![enter image description here](https://www.kdnuggets.com/wp-content/uploads/scalar-vector-matrix-tensor.jpg)
## -   What is one hot encoding?
One-Hot Encoding is essentially the representation of categorical variables as binary vectors. These categorical values are first mapped to integer values. Each integer value is then represented as a binary vector with all 0s (except for the index of the integer marked as 1).



(One-Hot Encoding, temel olarak kategorik değişkenlerin ikili vektörler olarak temsilidir. Bu kategorik değerler ilk önce tamsayı değerlere eşlenir. Her tamsayı değeri daha sonra tüm 0'ları olan bir ikili vektör olarak temsil edilir (1 olarak işaretlenen tamsayı indeksi hariç))
![enter image description here](https://mertmekatronik.com/uploads/images/2020/10/image_750x_5f8c85c715869.jpg)
## What is CUDA programming?
CUDA is a parallel computing platform and programming model developed by Nvidia for general computing on its own GPUs (graphics processing units). CUDA enables developers to speed up compute-intensive applications by harnessing the power of GPUs for the parallelizable part of the computation.


(CUDA , Nvidia tarafından kendi GPU'larında (grafik işlem birimleri) genel hesaplama için geliştirilmiş bir paralel hesaplama platformu ve programlama modelidir. CUDA, geliştiricilerin hesaplamanın paralelleştirilebilir kısmı için GPU'ların gücünden yararlanarak yoğun hesaplama gerektiren uygulamaları hızlandırmalarını sağlar.)

# Design Segmentation Model
## -   What is the difference between CNN and Fully CNN (FCNN) ?
FCNN(Fully Convolutional Neural Network), unlike the classic CNN, which use the Fully Connected layers after the Convolutional layers in the network, the FCNN can take input of arbitrary size. U-Net is also a network structure based on FCN. Fully convolutional training takes the whole M x M image and produces outputs for all subimages in a single ConvNet forward pass.

## -   What are the different layers on CNN ?
There are three types of layers that make up the CNN which are the convolutional layers, pooling layers, and fully-connected (FC) layers.

### **1. Convolutional Layer**
This layer is the first layer that is used to extract the various features from the input images. In this layer, the mathematical operation of convolution is performed between the input image and a filter of a particular size MxM.


(Bu katman, girdi görüntülerinden çeşitli özellikleri çıkarmak için kullanılan ilk katmandır. Bu katmanda, girdi görüntüsü ile belirli bir MxM boyutundaki bir filtre arasında matematiksel evrişim işlemi gerçekleştirilir.)

### **2. Pooling Layer**
In most cases, a Convolutional Layer is followed by a Pooling Layer. The primary aim of this layer is to decrease the size of the convolved feature map to reduce the computational costs. This is performed by decreasing the connections between layers and independently operates on each feature map. Depending upon method used, there are several types of Pooling operations.


(Çoğu durumda, bir Convolution Katmanı, bir Pooling Katmanı izler. Bu katmanın birincil amacı, hesaplama maliyetlerini azaltmak için convolved feature map'in boyutunu küçültmektir. Bu, katmanlar arasındaki bağlantıları azaltarak gerçekleştirilir ve her bir feature map üzerinde bağımsız olarak çalışır. Kullanılan yönteme bağlı olarak, birkaç tür Pooling işlemi vardır.

### **3. Fully Connected Layer**
The Fully Connected (FC) layer consists of the weights and biases along with the neurons and is used to connect the neurons between two different layers. These layers are usually placed before the output layer and form the last few layers of a CNN Architecture.

(Fully Connected (FC) katmanı, nöronlarla birlikte ağırlıklardan ve biase'lardan oluşur ve nöronları iki farklı katman arasında bağlamak için kullanılır. Bu katmanlar genellikle çıktı katmanından önce yerleştirilir ve bir CNN Mimarisinin son birkaç katmanını oluşturur.) 

![enter image description here](https://miro.medium.com/max/3288/1*uAeANQIOQPqWZnnuH-VEyw.jpeg)

## -What is activation function ?
An activation function is a function used in artificial neural networks which outputs a small value for small inputs, and a larger value if its inputs exceed a threshold. If the inputs are large enough, the activation function "fires", otherwise it does nothing. In other words, an activation function is like a gate that checks that an incoming value is greater than a critical number.
Many multi-layer neural networks end in a penultimate layer which outputs real-valued scores that are not conveniently scaled and which may be difficult to work with. Here the softmax is very useful because it converts the scores to a normalized probability distribution, which can be displayed to a user or used as input to other systems. For this reason it is usual to append a softmax function as the final layer of the neural network.


(Birçok çok katmanlı sinir ağı, uygun şekilde ölçeklendirilmemiş ve birlikte çalışması zor olabilecek gerçek değerli puanlar veren sondan bir önceki katmanda sona erer. Burada softmax çok kullanışlıdır çünkü puanları bir kullanıcıya gösterilebilen veya diğer sistemlere girdi olarak kullanılabilen normalleştirilmiş bir olasılık dağılımına dönüştürür . Bu nedenle, sinir ağının son katmanı olarak bir softmax işlevi eklemek olağandır.)


![enter image description here](https://themaverickmeerkat.com/img/softmax/sigmoid_plot.jpg)


# Train
## -   What is parameter and hyper-parameter in NN ?
A model parameter is a configuration variable that is internal to the model and whose value can be estimated from data. Parameters are key to machine learning algorithms. They are the part of the model that is learned from historical training data.  these are the coefficients of the model, and they are chosen by the model itself. It means that the algorithm, while learning, optimizes these coefficients (according to a given optimization strategy) and returns an array of parameters which minimize the error. 
**Hyperparameters**: these are elements that, differently from the previous ones, you need to set. Furthermore, the model will not update them according to the optimization strategy: your manual intervention will always be needed.
· **Number of hidden layers**
· **Learning rate**
· **Momentum**
· **Activation function**
· **Minibatch size**
· **Epochs**
· **Dropout**


## Validation Dataset
The sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters. The evaluation becomes more biased as skill on the validation dataset is incorporated into the model configuration. The validation set is used to evaluate a given model, but this is for frequent evaluation. Hence the model occasionally sees this data, but never does it “Learn” from this. We use the validation set results, and update higher level hyperparameters. So the validation set affects a model, but only indirectly. The validation set is also known as the Dev set or the Development set. This makes sense since this dataset helps during the “development” stage of the model.

 ## -   What is epoch?
 The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters. An epoch is comprised of one or more batches.

## -   What is batch?
The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters.

## - What is iteration?
For each complete epoch, we have several iterations. Iteration is the number of batches or steps through partitioned packets of the training data, needed to complete one epoch.
if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch. So then it will take x/y iterations to complete 1 epoch.

 ## -   What Is the Cost Function?
 A cost function is a measure of "how good" a neural network did with respect to it's given training sample and the expected output. It also may depend on variables such as weights and biases. A cost function is a single value, not a vector, because it rates how good the neural network did as a whole.

 (Maliyet fonksiyonu, verilen eğitim örneğine ve beklenen çıktıya göre bir sinir ağının "ne kadar iyi" olduğunun bir ölçüsüdür. Ayrıca ağırlıklar ve önyargılar gibi değişkenlere de bağlı olabilir. Maliyet fonksiyonu, bir vektör değil, tek bir değerdir, çünkü sinir ağının bir bütün olarak ne kadar iyi olduğunu değerlendirir.)

## -What is/are the purpose(s) of an optimizer in NN?
Optimizers are algorithms or methods used to change the attributes of the neural network such as weights and learning rate to reduce the losses. How you should change your weights or learning rates of your neural network to reduce the losses is defined by the optimizers you use. Optimization algorithms are responsible for reducing the losses and to provide the most accurate results possible.

(Optimize ediciler, kayıpları azaltmak için sinir ağının ağırlıklar ve öğrenme oranı gibi özelliklerini değiştirmek için kullanılan algoritmalar veya yöntemlerdir.Kayıpları azaltmak için sinir ağınızın ağırlıklarını veya öğrenme oranlarını nasıl değiştirmeniz gerektiği, kullandığınız optimize ediciler tarafından belirlenir. Optimizasyon algoritmaları, kayıpları azaltmaktan ve mümkün olan en doğru sonuçları sağlamaktan sorumludur.)

## -What is Batch Gradient Descent & Stochastic Gradient Descent?
In **Batch Gradient Descent**, all the training data is taken into consideration to take a single step. We take the average of the gradients of all the training examples and then use that mean gradient to update our parameters. So that’s just one step of gradient descent in one epoch.
In **Stochastic Gradient Descent** (SGD), we consider just one example at a time to take a single step. We do the following steps in **one epoch** for SGD:
1.  Take an example
2.  Feed it to Neural Network
3.  Calculate it’s gradient
4.  Use the gradient we calculated in step 3 to update the weights
5.  Repeat steps 1–4 for all the examples in training dataset

SGD can be used for larger datasets. It converges faster when the dataset is large as it causes updates to the parameters more frequently.
(SGD, daha büyük veri kümeleri için kullanılabilir. Parametrelerin daha sık güncellenmesine neden olduğu için veri kümesi büyük olduğunda daha hızlı yakınsar.)


## -   What is Backpropogation ? What is used for ?
Artificial neural networks use backpropagation as a learning algorithm to compute a gradient descent with respect to weights. Desired outputs are compared to achieved system outputs, and then the systems are tuned by adjusting connection weights to narrow the difference between the two as much as possible. The algorithm gets its name because the weights are updated backwards, from output towards input.
A neural network propagates the signal of the input data forward through its parameters towards the moment of decision, and then _backpropagates_ information about the error, in reverse through the network, so that it can alter the parameters. This happens step by step:
-   The network makes a guess about data, using its parameters
-   The network’s is measured with a loss function
-   The error is backpropagated to adjust the wrong-headed parameters

Backpropagation takes the error associated with a wrong guess by a neural network, and uses that error to adjust the neural network’s parameters in the direction of less error.

![enter image description here](https://machinelearningknowledge.ai/wp-content/uploads/2019/10/Backpropagation.gif)
