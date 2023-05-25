 import java.io.IOException;
    import java.util.StringTokenizer;

    import org.apache.hadoop.conf.Configuration;
    import org.apache.hadoop.fs.Path;
    import org.apache.hadoop.io.IntWritable;
    import org.apache.hadoop.io.Text;
    import org.apache.hadoop.mapreduce.Job;
    import org.apache.hadoop.mapreduce.Mapper;
    import org.apache.hadoop.mapreduce.Reducer;
    import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
    import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

    public class WordCount {

        public static void main(String[] args) throws Exception {
            Configuration conf = new Configuration();
            Job job = Job.getInstance(conf, "word count");
            job.setJarByClass(WordCount.class);
            job.setMapperClass(WordCountMapper.class);
            job.setCombinerClass(WordCountReducer.class);
            job.setReducerClass(WordCountReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(IntWritable.class);
            FileInputFormat.addInputPath(job, new Path(args[0]));
            FileOutputFormat.setOutputPath(job, new Path(args[1]));
            System.exit(job.waitForCompletion(true) ? 0 : 1);
        }

        public static class WordCountMapper extends Mapper<Object, Text, Text, IntWritable> {

            private final static IntWritable one = new IntWritable(1);
            private Text word = new Text();

            public void map(Object key, Text value, Context context) throws IOException,
                    InterruptedException {
                StringTokenizer itr = new StringTokenizer(value.toString());
                while (itr.hasMoreTokens()) {
                    word.set(itr.nextToken());
                    context.write(word, one);
                }
            }
        }

        public static class WordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
            private IntWritable result = new IntWritable();

            public void reduce(Text key, Iterable<IntWritable> values, Context context)
                    throws IOException, InterruptedException {
                int sum = 0;
                for (IntWritable val : values) {
                    sum += val.get();
                }
                result.set(sum);
                context.write(key, result);
            }
        }

    }









1.	Write a Scala program to check the largest number among three given integers.
	
	// taking three variables
	var a: Int = 70
	var b: Int = 40
	var c: Int = 100

	// condition_1
	if (a > b)
	{
		// condition_2
		if(a > c)
		{
			println("a is largest");
		}
		
		else
		{
			println("c is largest")
		}
	
	}
	
	else
	{
		
		// condition_3
		if(b > c)
		{
			println("b is largest")
		}
		
		else
		{
			println("c is largest")
		}
	}











2) Write a Scala program to reverse an array of integer values.

var nums1 = Array(1789, 2035, 1899, 1456, 2013) 
    println("Orginal array:")
    for ( x <- nums1) {
       print(s"${x}, ")        
     }           
    var result1= test(nums1)
    println("\nReversed array:")
    for ( x <- result1) {
       print(s"${x}, ")        
     }

def test(nums: Array[Int]): Array[Int] = {
    var temp1 = 0
    var temp2 = 0
    var index_position = 0
    var index_last_pos = nums.length - 1   
    while (index_position < index_last_pos) {
    temp1 = nums(index_position)
    temp2 = nums(index_last_pos)
    nums(index_position) = temp2
    nums(index_last_pos) = temp1
    index_position += 1
    index_last_pos -= 1
     }
    nums
}





3) Write a Scala code to merge two integer arrays into a third array


 var IntArray1 = Array(10,11,12,13,14,15)
        var IntArray2 = Array(20,21,22,23,24,25)
        var IntArray3 = new Array[Int](12)
        var count:Int=0
        var count1:Int=0
        
        // Merge IntArray1 and IntArray2 into IntArray3.
        while(count<12)
        {
            if(count<6)
            IntArray3(count)=IntArray1(count)
            else
            {
                IntArray3(count)=IntArray2(count1)
                count1=count1+1
            }
            count=count+1
        }
        
        println("Elements of merged array:")
        count=0
        while(count<12)
        {
            printf("%d ",IntArray3(count))
            count=count+1
        }
        




X = data.drop(['class'], axis=1)
y = data.drop(['sepal length',  'sepal width',  'petal length',  'petal width'], axis=1)
print(X)
print(y)
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
model.score(X_test,y_test)

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print(accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
print("Confusion matrix:")
print(cm)

dist.plot()
plt.show()

def get_confusion_matrix_values(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])

TP, FP, FN, TN = get_confusion_matrix_values(y_test, y_pred)
print("TP: ", TP)
print("FP: ", FP)
print("FN: ", FN)
print("TN: ", TN)

print("The Accuracy is ", (TP+TN)/(TP+TN+FP+FN))
print("The precision is ", TP/(TP+FP))
print("The recall is ", TP/(TP+FN))
