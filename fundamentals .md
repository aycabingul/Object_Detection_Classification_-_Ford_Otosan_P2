# Answers

# Project

## What is Machine Learning?

With machine learning, humans input data as well as the answers expected from the data, and the outcome of the rules. These rules can then be applied to new data to produce original answers. A machine-learning system is trained rather than explicitly programmed. It’s presented with many examples relevant to a task, and it finds statistical structure in these examples that eventually allows the system to come up with rules for automating the task.
(Makine Öğrenmesi, veri ve beklenen çıktıları kullanarak yeni verilere özgün cevaplar üretecek kuralları öğrenir. Makine öğrenmesi sistemleri elle programlanmak yerine **eğitilirler**. Göreve yönelik verilen örneklere bakarak istatistiksel örüntüleri ortaya çıkartıp görevi otomatikleştirecek kuralları ortaya koyar.)

## What is Unsupervised vs Supervised learning difference?

### Supervised learning;

It consists of learning to map input data to known targets (also called annotations), given a set of examples (often annotated by humans).
(Bir veri seti üzerindeki girdilerden bilinen çıktılara/hedeflere olan eşleştirmeyi öğrenmeyi kapsar.

### Unsupervised learning;

This branch of machine learning consists of finding interesting transformations of the input data without the help of any targets, for data visualization, data compression, or data denoising, or to better understand the correlations present in the data at hand.
( Makine öğrenmesinin bu kategorisinin amacı veriyi görselleştirmek, verileri sıkıştırmak, verilerdeki gürültüyü azaltmak ya da eldeki veriler arasındaki kolerasyonu anlamak için girdilerden herhangi bir bilinen hedefin yardımı olmaksızın amaca uygun dönüşümleri bulmaktır.)
The main distinction between the two approaches is the use of labeled datasets. To put it simply, supervised learning uses labeled input and output data, while an unsupervised learning algorithm does not.
(İki yaklaşım arasındaki temel ayrım, etiketli veri kümelerinin kullanılmasıdır. Basitçe söylemek gerekirse, denetimli öğrenme etiketli girdi ve çıktı verilerini kullanırken denetimsiz öğrenme algoritması kullanmaz.)

## What is Deep Learning?

Deep learning is a specific subfield of machine learning: a new take on learning representations from data that emphasizes learning successive layers of increasingly meaningful representations. The deep in **deep learning** isn’t a reference to any kind of deeper understanding achieved by the approach; rather, it stands for this idea of successive layers of representations. How many layers contribute to a model of the data is called the *depth* of the model.
(Derin öğrenme, birbirini takip eden katmanlarda veriler işlenirken giderek artan şekilde daha kullanışlı gösterimler elde edilebilen makine öğrenmesinin bir alt alanıdır. Derin derken kastedilen **derin öğrenmenin** birtakım derin bilgiler elde etmesi değil, birbirini takip eden gösterim katmanları ifade edilmektedir. Modeldeki katman modelin *derinliğini* oluşturmaktadır.)

## What is Neural Network (NN)?

Modern deep learning often involves tens or even hundreds of successive layers of representations and they’ve all learned automatically from exposure to training data. Meanwhile, other approaches to machine learning tend to focus on learning only one or two layers of representations of the data; hence, they’re sometimes called shallow learning. In deep learning, these layered representations are (almost always) learned via models called neural networks, structured in literal layers stacked on top of each other.

Artificial neural networks (ANNs) are comprised of node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network.
(Modern derin öğrenme modelleri, onlarca hatta yüzlerce birbirini takip eden katmanlar içermektedir. Oysa diğer makine öğrenme algoritmaları, genelde bir veya iki katmandan oluşur ki bazen *sığ öğrenme* olarak da adlandırılırlar. Bu katmanlı gösterim, **sinir ağı** diye adlandırılan ve birbirini takip eden katmanları olan model sayesinde öğrenilmektedir.

Yapay sinir ağları (YSA), bir girdi katmanı, bir veya daha fazla gizli katman ve bir çıktı katmanı içeren bir düğüm katmanından oluşur. Her düğüm veya yapay nöron diğerine bağlanır ve ilişkili bir ağırlık ve eşiğe sahiptir. Herhangi bir düğümün çıktısı belirtilen eşik değerinin üzerindeyse, o düğüm etkinleştirilir ve ağın bir sonraki katmanına veri gönderilir. Aksi takdirde, ağın bir sonraki katmanına hiçbir veri iletilmez.)

![https://miro.medium.com/max/1612/0*3rRcXavz8hmHXAFL.png](https://miro.medium.com/max/1612/0*3rRcXavz8hmHXAFL.png)

## What is Convolution Neural Network (CNN)?

A convolutional neural network (ConvNet / Convolutional neural networks -CNN) is a deep learning algorithm that can take an input image and separate various aspects/objects in the image.
Convolutional neural networks are deep neural networks that are mainly used to classify images (for example, to name what they see), cluster by similarity (photo search), and perform object recognition in scenes.

- - CNN learns the filters automatically without mentioning it explicitly. These filters help in extracting the right and relevant features from the input data.
-- CNN captures the **spatial features** from an image. Spatial features refer to the arrangement of pixels and the relationship between them in an image. They help us in identifying the object accurately, the location of an object, as well as its relation with other objects in an image

(Bir evrişimsel sinir ağı (ConvNet / Convolutional neural networks -CNN), bir girdi görüntüsünü alıp, görüntüdeki çeşitli görünüşleri/nesneleri birbirinden ayırabilen derin öğrenme algoritmasıdır.
Evrişimli sinir ağları, temel olarak görüntüleri sınıflandırmak (örneğin gördüklerini isimlendirmek), benzerlikle kümelemek (fotoğraf arama) ve sahnelerde nesne tanıma yapmak için kullanılan derin yapay sinir ağlarıdır.

- - CNN, filtreleri açıkça belirtmeden otomatik olarak öğrenir. Bu filtreler, giriş verilerinden doğru ve ilgili özelliklerin çıkarılmasına yardımcı olur.
-- CNN bir görüntüden

**uzamsal özellikleri**

yakalar . Uzamsal özellikler, bir görüntüdeki piksellerin düzenini ve aralarındaki ilişkiyi ifade eder. Nesneyi, bir nesnenin konumunu ve bir görüntüdeki diğer nesnelerle ilişkisini doğru bir şekilde tanımlamamıza yardımcı olurlar.)

![https://www.researchgate.net/publication/326152216/figure/fig2/AS:644307941879809@1530626392505/Deep-convolutional-neural-network-DCNN-architecture-A-schematic-diagram-of-AlexNet.png](https://www.researchgate.net/publication/326152216/figure/fig2/AS:644307941879809@1530626392505/Deep-convolutional-neural-network-DCNN-architecture-A-schematic-diagram-of-AlexNet.png)

## What is segmentation task in NN?

Image segmentation is the process of classifying each pixel in an image belonging to a certain class and hence can be thought of as a classification problem per pixel. semantic segmentation falls under supervised learning. There are two types of segmentation techniques;

1. **Semantic segmentation**: Semantic segmentation is the process of classifying each pixel belonging to a particular label. It doesn't different across different instances of the same object. For example if there are 2 cats in an image, semantic segmentation gives the same label to all the pixels of both cats

    ![https://wiki.tum.de/download/attachments/23561833/sms.png?version=1&modificationDate=1483619907233&api=v2](https://wiki.tum.de/download/attachments/23561833/sms.png?version=1&modificationDate=1483619907233&api=v2)

2. **Instance segmentation**: Instance segmentation differs from semantic segmentation in the sense that it gives a unique label to every instance of a particular object in the image. As can be seen in the image above all 3 dogs are assigned different colors i.e different labels. With semantic segmentation all of them would have been assigned the same color.

( Semantik Segmentasyon, belirli bir sınıfa ait bir görüntüdeki her bir pikseli sınıflandırma işlemidir ve bu nedenle piksel başına bir sınıflandırma problemi olarak düşünülebilir. Semantik segmentasyon denetimli öğrenme kapsamına girer. İki tür segmentasyon tekniği vardır;

    [https://www.researchgate.net/profile/Vinorth-Varatharasan/publication/339328277/figure/fig1/AS:864554888204291@1583137356688/Semantic-segmentation-left-and-Instance-segmentation-right-8.ppm](https://www.researchgate.net/profile/Vinorth-Varatharasan/publication/339328277/figure/fig1/AS:864554888204291@1583137356688/Semantic-segmentation-left-and-Instance-segmentation-right-8.ppm)

3. **Semantic segmentation**: Semantik segmentasyon, belirli bir etikete ait her pikseli sınıflandırma işlemidir. Aynı nesnenin farklı örneklerinde farklı değildir. Örneğin bir görüntüde 2 kedi varsa, anlamsal bölümleme her iki kedinin tüm piksellerine aynı etiketi verir.
4. **Instance segmentation**: Örnek bölümleme, görüntüdeki belirli bir nesnenin her örneğine benzersiz bir etiket vermesi anlamında anlamsal bölümlemeden farklıdır. Yukarıdaki resimde görüldüğü gibi 3 köpeğe de farklı renkler yani farklı etiketler atanmıştır. Anlamsal bölümleme ile hepsine aynı renk atanırdı.)

## What is classification task in NN?

Artificial Neural Network applications are generally used in prediction, classification, data association, data interpretation and data filtering. Artificial neural networks used for classification take on the task of classifying input values. Classification is the categorization process in which objects are defined, differentiated and data is understood based on the training set. Classification is a supervised learning technique with a training set and correctly defined observations.
All classification tasks depend upon labeled datasets; that is, humans must transfer their knowledge to the dataset for a neural network to learn the correlation between labels and data. This is known as supervised learning.

- Detect faces, identify people in images, recognize facial expressions (angry, joyful)
- Identify objects in images (stop signs, pedestrians, lane markers…)
- Recognize gestures in video
- Detect voices, identify speakers, transcribe speech to text, recognize sentiment in voices
- Classify text as spam (in emails), or fraudulent (in insurance claims); recognize sentiment in text (customer feedback)

(Yapay Sinir Ağları uygulamaları genellikle tahmin, sınıflandırma, veri ilişkilendirme, veri yorumlama ve veri filtreleme işlemlerinde kullanılmaktadır.  Sınıflandırma amacıyla kullanılan yapay sinir ağları, girdi değerlerini sınıflama görevini üstlenirler. Sınıflandırma, nesnelerin tanımlandığı, farklılaştığı ve verilerin eğitim seti temelinde anlaşıldığı kategorizasyon sürecidir. Sınıflandırma, bir eğitim seti ve doğru tanımlanmış gözlemlerin bulunduğu denetimli bir öğrenme tekniğidir.
Tüm sınıflandırma görevleri, etiketlenmiş veri kümelerine bağlıdır; yani, bir sinir ağının etiketler ve veriler arasındaki ilişkiyi öğrenmesi için insanların bilgilerini veri kümesine aktarması gerekir. Bu, denetimli öğrenme olarak bilinir .

- Yüzleri algılayın, görüntülerdeki insanları tanımlayın, yüz ifadelerini tanıyın (kızgın, neşeli)
- Görüntülerdeki nesneleri tanımlayın (dur işaretleri, yayalar, şerit işaretleri…)
- Videodaki hareketleri tanıma
- Sesleri algılayın, konuşmacıları tanımlayın, konuşmayı metne dönüştürün, seslerdeki duyguyu tanıyın
- Metni spam (e-postalarda) veya hileli (sigorta taleplerinde) olarak sınıflandırma; metindeki duyarlılığı tanıma (müşteri geri bildirimi).

    ![https://learnopencv.com/wp-content/uploads/2017/11/cnn-schema1.jpg](https://learnopencv.com/wp-content/uploads/2017/11/cnn-schema1.jpg)

## Compare segmentation and classification in NN.

The difference between segmentation and classification is clear at some extend. And there is one difference between both of them. The classification process is easier than segmentation, in classification, all objects in a single image are grouped or categorized into a single class. While in segmentation each object of a single class in an image is highlighted with different shades to make them recognizable to computer vision.
(Segmentasyon ve sınıflandırma arasındaki fark bir dereceye kadar açıktır. Ve ikisi arasında tek bir fark var. Sınıflandırma işlemi segmentasyondan daha kolaydır, sınıflandırmada tek bir görüntüdeki tüm nesneler tek bir sınıfa gruplanır veya kategorize edilir. Segmentasyon sırasında, bir görüntüdeki tek bir sınıfın her bir nesnesi, bilgisayarla görüye tanınabilmesi için farklı gölgelerle vurgulanır.)

## What is data and dataset difference?

- **Data** are observations or measurements (unprocessed or processed) represented as text, numbers, or multimedia.
- A **dataset** is a structured collection of data generally associated with a unique body of work.

## What is the difference between supervised and unsupervised learning in terms of dataset?

**In a supervised learning** model, the algorithm learns on a labeled **dataset**, providing an answer key that the algorithm can use to evaluate its accuracy on training data. An **unsupervised** model, in contrast, provides unlabeled data that the algorithm tries to make sense of by extracting features and patterns on its own.
(Denetimli bir öğrenme modelinde, algoritma etiketli bir veri kümesi üzerinde öğrenir ve algoritmanın eğitim verileri üzerindeki doğruluğunu değerlendirmek için kullanabileceği bir cevap anahtarı sağlar. Denetimsiz bir model, aksine, algoritmanın kendi başına özellikleri ve kalıpları çıkararak anlamlandırmaya çalıştığı etiketlenmemiş verileri sağlar.)

# Data Preprocessing

## Extracting Masks

### What is color space?

A

**color space**

is a coordinate system in which each color is represented as a single point. Colors are composed of a mixture of blue, green, and red colors because they react differently at different wavelengths.
(

**Renk uzayı**

, her rengin tek bir nokta olarak temsil edildiği bir koordinat sistemidir. Renkler, farklı dalga boylarında farklı tepkime verdikleri için mavi, yeşil ve kırmızı renklerin karışımından oluşur.)

![https://www.hisour.com/wp-content/uploads/2018/03/RGB-color-space.jpg](https://www.hisour.com/wp-content/uploads/2018/03/RGB-color-space.jpg)

### What RGB stands for?

**RGB** means Red Green Blue, ie the primary colors in additive color synthesis. An **RGB** file consists of composite layers of Red, Gree, and Blue, each being coded on 256 levels from 0 to 255.
(**RGB**, Kırmızı Yeşil Mavi anlamına gelir, yani katkılı renk sentezindeki ana renkler. Bir **RGB** dosyası, her biri 0 ila 255 arasında 256 düzeyde kodlanmış Kırmızı, Yeşil ve Mavi bileşik katmanlarından oluşur.)

### In Python, can we transform from one color space to another?

There are more than 150 color-space conversion methods available in OpenCV. A popular computer vision library written in C/C++ with bindings for Python, OpenCV provides easy ways of manipulating color spaces.

(OpenCV'de 150'den fazla renk alanı dönüştürme yöntemi mevcuttur.
C/C++ ile Python için binding'lerle yazılmış popüler bir bilgisayarlı vision kütüphanesi olan OpenCV, renk uzaylarını değiştirmenin kolay yollarını sağlar.)

### What is the popular library for image processing?

OpenCV is one of the most famous and widely used open-source **libraries** for computer vision tasks such as **image processing**, object detection, face detection, **image** segmentation, face recognition, and many more. Other than this, it can also be used for machine learning tasks.
(OpenCV, görüntü işleme, nesne algılama, yüz algılama, görüntü bölütleme, yüz tanıma ve daha pek çok bilgisayarla görme görevleri için en ünlü ve yaygın olarak kullanılan açık kaynak kütüphanelerinden biridir. Bunun dışında makine öğrenimi görevleri için de kullanılabilir.)

# Converting into Tensor

## What is Computational Graph?

A computational graph is a directed graph where the nodes correspond to

**operations**

or

**variables**

. Variables can feed their value into operations, and operations can feed their output into other operations. This way, every node in the graph defines a function of the variables.
(Bir hesaplama grafiği, düğümlerin işlemlere veya değişkenlere karşılık geldiği yönlendirilmiş bir grafiktir. Değişkenler değerlerini işlemlere besleyebilir ve işlemler çıktılarını diğer işlemlere besleyebilir. Bu şekilde, grafikteki her düğüm değişkenlerin bir fonksiyonunu tanımlar.)

![http://media5.datahacker.rs/2021/01/54-1.jpg](http://media5.datahacker.rs/2021/01/54-1.jpg)

## What is Tensor?

The values that are fed into the nodes and come out of the nodes are called

**tensors**

, which is just a fancy word for a multi-dimensional array. Hence, it subsumes scalars, vectors, and matrices as well as tensors of a higher rank.
(Düğümlere beslenen ve düğümlerden çıkan değerlere

**tensörler**

denir, bu sadece çok boyutlu bir dizi için süslü bir kelimedir.)

![https://www.kdnuggets.com/wp-content/uploads/scalar-vector-matrix-tensor.jpg](https://www.kdnuggets.com/wp-content/uploads/scalar-vector-matrix-tensor.jpg)

## What is one hot encoding?

One-Hot Encoding is essentially the representation of categorical variables as binary vectors. These categorical values are first mapped to integer values. Each integer value is then represented as a binary vector with all 0s (except for the index of the integer marked as 1).
(One-Hot Encoding, temel olarak kategorik değişkenlerin ikili vektörler olarak temsilidir. Bu kategorik değerler ilk önce tamsayı değerlere eşlenir. Her tamsayı değeri daha sonra tüm 0'ları olan bir ikili vektör olarak temsil edilir (1 olarak işaretlenen tamsayı indeksi hariç))

![https://mertmekatronik.com/uploads/images/2020/10/image_750x_5f8c85c715869.jpg](https://mertmekatronik.com/uploads/images/2020/10/image_750x_5f8c85c715869.jpg)

## What is CUDA programming?

CUDA is a parallel computing platform and programming model developed by Nvidia for general computing on its own GPUs (graphics processing units). CUDA enables developers to speed up compute-intensive applications by harnessing the power of GPUs for the parallelizable part of the computation.
(CUDA , Nvidia tarafından kendi GPU'larında (grafik işlem birimleri) genel hesaplama için geliştirilmiş bir paralel hesaplama platformu ve programlama modelidir. CUDA, geliştiricilerin hesaplamanın paralelleştirilebilir kısmı için GPU'ların gücünden yararlanarak yoğun hesaplama gerektiren uygulamaları hızlandırmalarını sağlar.)

# Design Segmentation Model

## What is the difference between CNN and Fully CNN (FCNN)?

FCNN(Fully Convolutional Neural Network), unlike the classic CNN, which uses the Fully Connected layers after the Convolutional layers in the network, the FCNN can take input of arbitrary size. U-Net is also a network structure based on FCN.

# Object Recognition

## What is an image classification task?

It involves the extraction of information from an image and then associating the extracted information to one or more class labels. Image classification within the machine learning domain can be approached as a supervised learning task.

Image classification, at its very core, is the task of assigning a label to an image from a predefined set of categories.
Practically, this means that our task is to analyze an input image and return a label that categorizes the image. The label is always from a predefined set of possible categories.

For example, let’s assume that our set of possible categories includes:

![Answers%20d9fde1dc30ef4cff90a55802c20d2339/Untitled.png](Answers%20d9fde1dc30ef4cff90a55802c20d2339/Untitled.png)

Then we present the following image  to our classification system:

![Answers%20d9fde1dc30ef4cff90a55802c20d2339/Untitled%201.png](Answers%20d9fde1dc30ef4cff90a55802c20d2339/Untitled%201.png)

Our goal here is to take this input image and assign a label to it from our **categories set** — in this case, **dog**.

Our classification system could also assign multiple labels to the image via probabilities, such as dog: 95%; cat: 4%; panda: 1%.

More formally, given our input image of W×H pixels with three channels, Red, Green, and Blue, respectively, our goal is to take the W×H×3 = N pixel image and figure out how to correctly classify the contents of the image.

- **Image Classification**: Predict the type or class of an object in an image.
    - *Input*: An image with a single object, such as a photograph.
    - *Output*: A class label (e.g. one or more integers that are mapped to class labels).

## What is an object localization task?

Image classification involves predicting the class of one object in an image. Object localization refers to identifying the location of one or more objects in an image and drawing abounding boxes around their extent. Object detection combines these two tasks and localizes and classifies one or more objects in an image.

(Image classification(Görüntü sınıflandırma), bir görüntüdeki bir nesnenin sınıfını tahmin etmeyi içerir. Object localization(Nesne yerelleştirme), bir görüntüdeki bir veya daha fazla nesnenin konumunu belirleme ve boyutlarının etrafına bol miktarda kutu çizme anlamına gelir. Object detection(Nesne algılama) bu iki görevi birleştirir ve bir görüntüdeki bir veya daha fazla nesneyi yerelleştirir ve sınıflandırır.)

The task of object localization is to predict the object in an image as well as its boundaries. The difference between object localization and object detection is subtle. Simply, object localization aims to locate the main (or most visible) object in an image while object detection tries to find out all the objects and their boundaries.

(Nesne lokalizasyonunun görevi, bir görüntüdeki nesneyi ve sınırlarını tahmin etmektir. Nesne yerelleştirme ve nesne algılama arasındaki fark çok incedir. Basitçe, nesne yerelleştirme bir görüntüdeki ana (veya en görünür) nesneyi bulmayı amaçlarken nesne algılama tüm nesneleri ve sınırlarını bulmaya çalışır.)

- **Object Localization**: Locate the presence of objects in an image and indicate their location with a bounding box.
    - *Input*: An image with one or more objects, such as a photograph.
    - *Output*: One or more bounding boxes (e.g. defined by a point, width, and height).

## What is an object detection task?

Object detection is a computer vision technique that identifies and locates objects within an image or video. Specifically, object detection draws bounding boxes around these detected objects, which allow us to locate where said objects are in (or how they move through) a given scene.

Object detection is commonly confused with image recognition, so before we proceed, we must clarify the distinctions between them.

Image recognition assigns a label to an image. A picture of a dog receives the label “dog”. A picture of two dogs still receives the label “dog”. On the other hand, object detection draws a box around each dog and labels the box “dog”. The model predicts where each object is and what label should be applied. In that way, object detection provides more information about an image than recognition.

- **Object Detection**: Locate the presence of objects with a bounding box and types or classes of the located objects in an image.
    - *Input*: An image with one or more objects, such as a photograph.
    - *Output*: One or more bounding boxes (e.g. defined by a point, width, and height), and a class label for each bounding box.

![Answers%20d9fde1dc30ef4cff90a55802c20d2339/Untitled%202.png](Answers%20d9fde1dc30ef4cff90a55802c20d2339/Untitled%202.png)

![Answers%20d9fde1dc30ef4cff90a55802c20d2339/Untitled%203.png](Answers%20d9fde1dc30ef4cff90a55802c20d2339/Untitled%203.png)

## What is an object recognition task?

Object recognition is a general term to describe a collection of related computer vision tasks that involve identifying objects in digital photographs.

We will be using the term object recognition broadly to encompass both image classification (a task requiring an algorithm to determine what object classes are present in the image) as well as to object detection (a task requiring an algorithm to localize all objects present in the image.)

Object Recognition is responding to the question "What is the object in the image" Whereas, Object detection is answering the question "Where is that object"?

(Nesne Tanıma, "Görüntüdeki nesne nedir" sorusuna yanıt verirken Nesne algılama, "O nesne nerede" sorusuna yanıt verir)

- *input*: an image containing unknown object(s)

    Possibly, the position of the object can be marked in the input, or the input might be only a clear image of (not-occluded) object.

- *output*: position(s) and label(s) (names) of the objects in the image.

![Answers%20d9fde1dc30ef4cff90a55802c20d2339/Untitled%204.png](Answers%20d9fde1dc30ef4cff90a55802c20d2339/Untitled%204.png)

## What is bounding box regression?

Bounding-box regression is a popular technique to refine or predict localization boxes in recent object detection approaches. Typically, bounding-box regressors are trained to regress from either region proposals or fixed anchor boxes to nearby bounding boxes of predefined target object classes.

## What is non-max suppression?

Non Maximum Suppression (NMS) is a technique used in many computer vision algorithms. It is a class of algorithms to select one entity (e.g. bounding boxes) out of many overlapping entities. The selection criteria can be chosen to arrive at particular results. Most commonly, the criteria are some form of probability number along with some form of overlap measure (e.g. IOU).

Most object detection algorithms use NMS to whittle down a large number of detected rectangles to a few. At the most basic level, most object detectors do some form of windowing. Many, thousands, windows of various sizes and shapes are generated either directly on the image or on a feature of the image. These windows supposedly contain only one object, and a classifier is used to obtain a probability/score for each class. Once the detector outputs a large number of bounding boxes, it is necessary to pick the best ones. NMS is the most commonly used algorithm for this task. In essence, it is a form of clustering algorithm.

(Çoğu nesne algılama algoritması, çok sayıda algılanan dikdörtgeni birkaç taneye indirgemek için NMS kullanır. En temel düzeyde, çoğu nesne dedektörü bir tür pencereleme yapar. Çeşitli boyut ve şekillerde çok sayıda, binlerce pencere ya doğrudan görüntü üzerinde ya da görüntünün bir özelliği üzerinde oluşturulur. Bu pencerelerin yalnızca bir nesne içerdiği varsayılır ve her sınıf için bir olasılık/puan elde etmek için bir sınıflandırıcı kullanılır. Dedektör çok sayıda sınırlayıcı kutu çıkardığında, en iyilerini seçmek gerekir. NMS, bu görev için en yaygın kullanılan algoritmadır. Özünde, kümeleme algoritmasının bir şeklidir.)

![Answers%20d9fde1dc30ef4cff90a55802c20d2339/Untitled%205.png](Answers%20d9fde1dc30ef4cff90a55802c20d2339/Untitled%205.png)