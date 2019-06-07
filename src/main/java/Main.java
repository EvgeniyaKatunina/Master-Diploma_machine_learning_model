import com.univocity.parsers.tsv.TsvParser;
import com.univocity.parsers.tsv.TsvParserSettings;
import com.univocity.parsers.tsv.TsvWriter;
import com.univocity.parsers.tsv.TsvWriterSettings;
import mloss.roc.Curve;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import smile.classification.NeuralNetwork;
import smile.regression.RandomForest;
import smile.regression.RidgeRegression;
import smile.validation.Precision;
import smile.validation.Validation;

import java.io.*;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.*;
import java.util.function.ObjDoubleConsumer;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;

public class Main {

    public static void main(String[] args) throws IOException {
        TsvParserSettings settings = new TsvParserSettings();
        settings.getFormat().setLineSeparator("\n");
        settings.setMaxCharsPerColumn(15000);
        TsvParser parser = new TsvParser(settings);
        List<String[]> usersScoreRows = parser.parseAll(new BufferedReader(new FileReader("users_score.tsv")));
        double nominator = 0;
        double denominator1 = 0;
        double denominator2 = 0;
        for (int i = 1; i < usersScoreRows.size(); i++) {
            String[] row = usersScoreRows.get(i);
            double a = Double.parseDouble(row[1]);
            double b = Double.parseDouble(row[2]);
            nominator += (a * b);
            denominator1 += (a * a);
            denominator2 += (b * b);
        }
        double similarity = nominator / (Math.sqrt(denominator1) * Math.sqrt(denominator2));
        nominator = 0;
        denominator1 = 0;
        denominator2 = 0;
        System.out.println("similarity = " + similarity);
        for (int i = 1; i < usersScoreRows.size(); i++) {
            String[] row = usersScoreRows.get(i);
            int tasksNumber = Integer.parseInt(row[3]);
            if (tasksNumber > 100) {
                double a = Double.parseDouble(row[1]);
                double b = Double.parseDouble(row[2]);
                nominator += (a * b);
                denominator1 += (a * a);
                denominator2 += (b * b);
            }
        }
        similarity = nominator / (Math.sqrt(denominator1) * Math.sqrt(denominator2));
        System.out.println("similarity 100+ = " + similarity);
        double rmse = 0;
        for (int i = 1; i < usersScoreRows.size(); i++) {
            String[] row = usersScoreRows.get(i);
            int tasksNumber = Integer.parseInt(row[3]);
            if (tasksNumber > 100) {
                double a = Double.parseDouble(row[1]);
                double b = Double.parseDouble(row[2]);
                rmse += (a - b) * (a - b);
            }
        }
        rmse = Math.sqrt(rmse / (usersScoreRows.size() - 1));
        System.out.println("rmse +100 = " + rmse);

        rmse = 0;

        for (int i = 1; i < usersScoreRows.size(); i++) {
            String[] row = usersScoreRows.get(i);
            double a = Double.parseDouble(row[1]);
            double b = Double.parseDouble(row[2]);
            rmse += (a - b) * (a - b);
        }
        rmse = Math.sqrt(rmse / (usersScoreRows.size() - 1));
        System.out.println("rmse  = " + rmse);

        List<String[]> galResults = parser.parseAll(new BufferedReader(new FileReader("object-probabilities.txt")));
        double overallProbability = 0;
        double overallProbabilityLog = 0;
        for (int i = 1; i < galResults.size(); i++) {
            String[] row = galResults.get(i);
            double max = 0;
            for (int j = 6; j < 10; j++) {
                double prob = Double.parseDouble(row[j]);
                if (prob > max) {
                    max = prob;
                }
            }
            overallProbability += max;
            overallProbabilityLog += Math.log(max);
        }
        System.out.println("log quality = " + overallProbabilityLog);
        System.out.println(overallProbability / (galResults.size() - 1));
        List<String[]> allRows = parser.parseAll(new BufferedReader(new FileReader("assignments_changed.tsv")));
        List<String[]> allRowsAggregated = parser.parseAll(new BufferedReader(new FileReader(
                "aggregated_results_pool_1036853__2017_12_01.tsv")));
        HashMap<Integer, List<ObjectDescription>> taskId2Descriptions = new HashMap<>();
        HashMap<Integer, List<ObjectDescription>> userId2Descriptions = new HashMap<>();
        HashMap<Integer, Long> userId2SessionStart = new HashMap<>();
        for (int i = 1; i < allRows.size(); i++) {
            String[] row = allRows.get(i);
            String activity = row[6];
            if (activity == null) {
                continue;
            }
            String[] actions = activity.split("],\\[");
            if (actions.length <= 2) {
                continue;
            }
            String answerPrefix = "answer\"\":\"\"";
            String click = "click";
            int answerChanges = 0;
            long clickTime = 0;
            HashSet<Integer> answers = new HashSet<>();
            long[] actionTimes = new long[actions.length - 1];
            for (int j = 1; j < actions.length; j++) {
                String actionTime = actions[j].substring(2, actions[j].substring(2).indexOf("\""));
                actionTimes[j - 1] = LocalDateTime.parse(actionTime).toEpochSecond(ZoneOffset.UTC);
            }
            for (int j = 1; j < actions.length; j++) {
                String action = actions[j];
                if (action.contains(answerPrefix)) {
                    answerChanges++;
                    int answerIndex = action.indexOf(answerPrefix) + answerPrefix.length();
                    try {
                        int answer = Integer.parseInt(action.substring(answerIndex, answerIndex + 1));
                        answers.add(answer);
                    } catch (NumberFormatException e) {
                        continue;
                    }
                }
                if (action.contains(click)) {
                    clickTime = actionTimes[j - 1];
                }
            }
            if (answerChanges == 0 || answers.size() == 0 || clickTime == 0) {
                continue;
            }
            ObjectDescription currentObj = new ObjectDescription();
            currentObj.answer = Integer.parseInt(row[7]);
            currentObj.taskId = Integer.parseInt(row[0]);
            currentObj.userId = Integer.parseInt(row[13]);
            long startTime = actionTimes[0];
            long finishTime = actionTimes[actionTimes.length - 1];
            long answerTime = finishTime - startTime;
            if (answerTime > 30) {
                currentObj.answerTime = 30;
            } else {
                currentObj.answerTime = answerTime;
            }
            if (!userId2SessionStart.containsKey(currentObj.userId)) {
                userId2SessionStart.put(currentObj.userId, startTime);
            }
            currentObj.timeFromStartOfSession = startTime - userId2SessionStart.get(currentObj.userId);
            long idlenessTime = 0;
            long maxIdleness = 20;
            for (int j = 1; j < actionTimes.length; j++) {
                long diff = actionTimes[j] - actionTimes[j - 1];
                if (diff > maxIdleness) {
                    idlenessTime += (diff - maxIdleness);
                }
            }
            if (idlenessTime > 20) {
                idlenessTime = 20;
            }
            currentObj.idlenessTime = idlenessTime;
            if (answerChanges > 26) {
                currentObj.answerChanges = 13;
            } else {
                currentObj.answerChanges = answerChanges / 2;
            }
            currentObj.answerVariants = answers.size();
            long timeForApproval = finishTime - clickTime;
            if (timeForApproval > 12) {
                timeForApproval = 12;
            }
            currentObj.timeForAnswerApproval = timeForApproval;
            taskId2Descriptions.computeIfAbsent(currentObj.taskId, x -> new ArrayList<>()).add(currentObj);
            userId2Descriptions.computeIfAbsent(currentObj.userId, x -> new ArrayList<>()).add(currentObj);
        }
        List<String[]> workersErrorRatesRows = parser.parseAll(new BufferedReader(new FileReader("workers_error_rates"
                + ".tsv")));
        for (int i = 1; i < workersErrorRatesRows.size(); i++) {
            String[] row = workersErrorRatesRows.get(i);
            userId2Descriptions.get(Integer.parseInt(row[0])).forEach(objectDescription -> {
                objectDescription.dawidSkeneErrorRate = Double.parseDouble(row[1]);
            });
        }
        System.out.println("tasks number = " + taskId2Descriptions.keySet().size());
        System.out.println("users number = " + userId2Descriptions.keySet().size());
        System.out.println("average tasks per user = " + userId2Descriptions.values().stream().mapToInt(List::size).average().getAsDouble());
        taskId2Descriptions.keySet().forEach(taskId -> {
            double meanTimeForTask =
                    taskId2Descriptions.get(taskId).stream().mapToDouble(description -> description.answerTime).average().getAsDouble();
            taskId2Descriptions.get(taskId).forEach(description -> description.meanTimeForTask = meanTimeForTask);
        });
        userId2Descriptions.keySet().forEach(userId -> {
            double meanTimeForUser =
                    userId2Descriptions.get(userId).stream().mapToDouble(description -> description.answerTime).average().getAsDouble();
            userId2Descriptions.get(userId).forEach(description -> description.meanTimeForUser = meanTimeForUser);
        });

        HashMap<Integer, Integer> taskId2CorrectAnswer = new HashMap<>();
        for (int i = 1; i < allRowsAggregated.size(); i++) {
            String[] row = allRowsAggregated.get(i);
            taskId2CorrectAnswer.put(Integer.parseInt(row[0]), Integer.parseInt(row[6]));
        }
        List<ObjectDescription> descriptions =
                userId2Descriptions.values().stream().flatMap(Collection::stream).collect(Collectors.toList());

        TsvWriter majorityWriter = new TsvWriter(new PrintWriter(new FileWriter("majority_aggregated" + ".tsv")),
                new TsvWriterSettings());
        majorityWriter.writeHeaders("task_id", "answer", "confidence");
        ArrayList<String[]> majorityRows = new ArrayList<>();
        taskId2Descriptions.keySet().forEach(taskId -> {
            int correctAnswer = taskId2CorrectAnswer.get(taskId);
            int totalVote = 0;
            int correctVotes = 0;
            List<ObjectDescription> answers = taskId2Descriptions.get(taskId);
            for (ObjectDescription answer : answers) {
                if (answer.answer == correctAnswer) {
                    correctVotes++;
                }
                totalVote++;
            }
            double confidence = (double) correctVotes / totalVote;
            majorityRows.add(new String[]{Integer.toString(taskId), Integer.toString(correctAnswer),
                    Double.toString(confidence)});
        });
        double majorityVoteQuality =
                majorityRows.stream().mapToDouble(s -> Double.parseDouble(s[2])).average().getAsDouble();
        majorityWriter.writeRowsAndClose(majorityRows.toArray(new String[majorityRows.size()][3]));
        System.out.println("majority vote quality = " + majorityVoteQuality);


        Histograms.drawHistogram(descriptions.stream().map(d -> d.answerTime).collect(Collectors.toList()),
                "answer " + "time", 30, 0);

        Histograms.drawHistogram(descriptions.stream().map(d -> d.answerChanges).collect(Collectors.toList()),
                "answer changes", 13, 0);

        Histograms.drawHistogram(descriptions.stream().map(d -> d.answerVariants).collect(Collectors.toList()),
                "answer variants", 2, 0);

        Histograms.drawHistogram(descriptions.stream().map(d -> d.meanTimeForUser).collect(Collectors.toList()),
                "mean time for user", 30, 0);

        Histograms.drawHistogram(descriptions.stream().map(d -> d.meanTimeForTask).collect(Collectors.toList()),
                "mean time for task", 30, 0);

        Histograms.drawHistogram(descriptions.stream().map(d -> d.timeFromStartOfSession).collect(Collectors.toList()), "time from start of session", 30, 0);

        Histograms.drawHistogram(descriptions.stream().map(d -> d.idlenessTime).collect(Collectors.toList()),
                "idleness time", 30, 0);

        Histograms.drawHistogram(descriptions.stream().map(d -> d.timeForAnswerApproval).collect(Collectors.toList())
                , "time for approval", 30, 0);

        Histograms.drawHistogram(userId2Descriptions.values().stream().map(x -> (double) x.size()).collect(Collectors.toList()), "tasks by user", 30, 0);

        Histograms.drawHistogram(descriptions.stream().map(d -> d.dawidSkeneErrorRate).collect(Collectors.toList()),
                "users error rate", 3, 0);

        normalize(descriptions, d -> d.meanTimeForUser, (d, arg) -> d.meanTimeForUser = arg);
        normalize(descriptions, d -> d.answerTime, (d, arg) -> d.answerTime = arg);
        normalize(descriptions, d -> d.meanTimeForTask, (d, arg) -> d.meanTimeForTask = arg);
        normalize(descriptions, d -> d.timeForAnswerApproval, (d, arg) -> d.timeForAnswerApproval = arg);
        normalize(descriptions, d -> d.answerVariants, (d, arg) -> d.answerVariants = arg);
        normalize(descriptions, d -> d.answerChanges, (d, arg) -> d.answerChanges = arg);
        normalize(descriptions, d -> d.idlenessTime, (d, arg) -> d.idlenessTime = arg);
        normalize(descriptions, d -> d.timeFromStartOfSession, (d, arg) -> d.timeFromStartOfSession = arg);
        normalize(descriptions, d -> d.dawidSkeneErrorRate, (d, arg) -> d.dawidSkeneErrorRate = arg);

        ArrayList<Integer> hasWrongAnswers = new ArrayList<>();
        ArrayList<Integer> onlyCorrectAnswers = new ArrayList<>();
        taskId2Descriptions.keySet().forEach(taskId -> {
            int correctAnswer = taskId2CorrectAnswer.get(taskId);
            boolean wrongAnswersPresent =
                    taskId2Descriptions.get(taskId).stream().anyMatch(description -> description.answer != correctAnswer);
            if (wrongAnswersPresent) {
                hasWrongAnswers.add(taskId);
            } else {
                onlyCorrectAnswers.add(taskId);
            }
        });
        HashMap<Integer, ArrayList<Double>> userId2Score = new HashMap<>();
        HashMap<Integer, ArrayList<Integer>> userId2CorrectPart = new HashMap<>();
        userId2Descriptions.keySet().forEach(userId -> {
            List<ObjectDescription> userDescriptions = userId2Descriptions.get(userId);
            for (ObjectDescription description : userDescriptions) {
                userId2CorrectPart.computeIfAbsent(userId, x -> new ArrayList<>()).add(description.answer == taskId2CorrectAnswer.get(description.taskId) ? 1 : 0);
            }
        });

        System.out.println(descriptions.stream().filter(d -> d.answer != taskId2CorrectAnswer.get(d.taskId)).count());
        TsvWriter writerTotalResults = new TsvWriter(new PrintWriter(new FileWriter("total_answer_statistics")),
                new TsvWriterSettings());
        writerTotalResults.writeHeaders("taskId", "userId", "answer", "confidence");
        ArrayList<String[]> totalResultsRows = new ArrayList<>();

        int[] truth = new int[9572];
        double[] positivePredicitions = new double[9572];
        int counterROC = 0;

        ArrayList<String[]> majorityWeightedRows = new ArrayList<>();
        int folds = 10;
        int wrongAnswersPerIteration = hasWrongAnswers.size() / folds;
        int correctAnswersPerIteration = onlyCorrectAnswers.size() / folds;
        for (int i = 0; i < folds; i++) {
            ArrayList<ObjectDescription> wrongAnswers = new ArrayList<>();
            int wrongAnswerStart = wrongAnswersPerIteration * i;
            int wrongAnswerEnd = wrongAnswerStart + wrongAnswersPerIteration;
            int correctAnswerStart = correctAnswersPerIteration * i;
            int correctAnswerEnd = correctAnswerStart + correctAnswersPerIteration;
            ArrayList<List<Double>> featuresToPredictList = new ArrayList<>();
            ArrayList<List<Double>> trainFeaturesList = new ArrayList<>();
            ArrayList<Double> answers = new ArrayList<>();
            ArrayList<ObjectDescription> objectsToPredict = new ArrayList<>();
            for (int j = 0; j < hasWrongAnswers.size(); j++) {
                int taskId = hasWrongAnswers.get(j);
                List<ObjectDescription> objectDescriptions = taskId2Descriptions.get(taskId);
                if (j >= wrongAnswerStart && j < wrongAnswerEnd) {
                    createList(featuresToPredictList, objectDescriptions);
                    objectsToPredict.addAll(objectDescriptions);
                } else {
                    createList(trainFeaturesList, objectDescriptions);
                    int correctAnswer = taskId2CorrectAnswer.get(taskId);
                    for (int k = 0; k < objectDescriptions.size(); k++) {
                        answers.add((double) (correctAnswer == objectDescriptions.get(k).answer ? 1 : 0));
                    }
                    objectDescriptions.forEach(description -> {
                        if (description.answer != taskId2CorrectAnswer.get(taskId)) {
                            wrongAnswers.add(description);
                        }
                    });
                }
            }
            for (int j = 0; j < onlyCorrectAnswers.size(); j++) {
                int taskId = onlyCorrectAnswers.get(j);
                List<ObjectDescription> objectDescriptions = taskId2Descriptions.get(taskId);
                if (j >= correctAnswerStart && j < correctAnswerEnd) {
                    createList(featuresToPredictList, objectDescriptions);
                    objectsToPredict.addAll(objectDescriptions);
                } else {
                    createList(trainFeaturesList, objectDescriptions);
                    int correctAnswer = taskId2CorrectAnswer.get(taskId);
                    for (int k = 0; k < objectDescriptions.size(); k++) {
                        answers.add((double) (correctAnswer == objectDescriptions.get(k).answer ? 1 : 0));
                    }
                }
            }
            int wrongAnswersAdded = wrongAnswers.size();
            int answersAdded = trainFeaturesList.size();
            ArrayList<ObjectDescription> wrongAnswersToAdd = new ArrayList<>();
            for (int j = 0; j < (answersAdded - wrongAnswersAdded); j++) {
                Random r = new Random();
                int randomIndex = r.nextInt(wrongAnswers.size());
                ObjectDescription description = wrongAnswers.get(randomIndex);
                wrongAnswersToAdd.add(description);
                answers.add(0.0);
            }
            createList(trainFeaturesList, wrongAnswersToAdd);
            double[][] featuresToPredict = new double[featuresToPredictList.size()][9];
            double[][] trainFeatures = new double[trainFeaturesList.size()][9];
            double[] answersArray = new double[answers.size()];
            for (int j = 0; j < answers.size(); j++) {
                answersArray[j] = answers.get(j);
            }
            for (int j = 0; j < featuresToPredictList.size(); j++) {
                List<Double> features = featuresToPredictList.get(j);
                for (int k = 0; k < features.size(); k++) {
                    featuresToPredict[j][k] = features.get(k);
                }
            }
            for (int j = 0; j < trainFeaturesList.size(); j++) {
                List<Double> features = trainFeaturesList.get(j);
                for (int k = 0; k < features.size(); k++) {
                    trainFeatures[j][k] = features.get(k);
                }
            }
            RandomForest rf = new RandomForest(trainFeatures, answersArray, 200);
            NeuralNetwork nn = new NeuralNetwork(NeuralNetwork.ErrorFunction.LEAST_MEAN_SQUARES, 8, 8, 8, 8, 8);
            RidgeRegression rr = new RidgeRegression(trainFeatures, answersArray, 287);
            double[] importance = rf.importance();
            int[] answersArrayInt = new int[answersArray.length];
            for (int j = 0; j < answersArray.length; j++) {
                answersArrayInt[j] = (int) answersArray[j];
            }
            assert trainFeatures.length == answersArrayInt.length;
//            nn.learn(trainFeatures, answersArrayInt);
            ArrayList<String[]> crossValidatedRows = new ArrayList<>();
            TsvWriter writerResults = new TsvWriter(new PrintWriter(new FileWriter("cross_validation" + i + ".tsv")),
                    new TsvWriterSettings());
            writerResults.writeHeaders("taskId", "userId", "answer", "confidence");
            for (int j = 0; j < featuresToPredict.length; j++) {
                ObjectDescription objectDescription = objectsToPredict.get(j);
                double score = rf.predict(featuresToPredict[j]);
                userId2Score.computeIfAbsent(objectDescription.userId, x -> new ArrayList<>()).add(score);
                String[] row = Arrays.asList(Integer.toString(objectDescription.taskId),
                        Integer.toString(objectDescription.userId), Integer.toString(objectDescription.answer),
                        String.format(Locale.US, "%.2f", score)).toArray(new String[4]);
                truth[counterROC] = objectDescription.answer == taskId2CorrectAnswer.get(objectDescription.taskId) ?
                        1 : 0;
                positivePredicitions[counterROC] = truth[counterROC] == 1 ? score : 1 - score;
                counterROC++;
                crossValidatedRows.add(row);
            }
            int currentTask = Integer.parseInt(crossValidatedRows.get(0)[0]);
            double correctVotes = 0;
            double totalVotes = 0;
            for (int k = 0; k < crossValidatedRows.size(); k++) {
                String[] row = crossValidatedRows.get(k);
                int taskId = Integer.parseInt(row[0]);
                if (currentTask != taskId) {
                    majorityWeightedRows.add(new String[]{Integer.toString(currentTask),
                            Integer.toString(taskId2CorrectAnswer.get(currentTask)),
                            Double.toString(correctVotes / totalVotes)});
                    currentTask = taskId;
                    correctVotes = 0;
                    totalVotes = 0;
                }
                int correctAnswer = taskId2CorrectAnswer.get(taskId);
                int currentAnswer = Integer.parseInt(row[2]);
                double weight = Double.parseDouble(row[3]);
                if (correctAnswer == currentAnswer) {
                    correctVotes += weight;
                }
                totalVotes += weight;
            }
            majorityWeightedRows.add(new String[]{Integer.toString(currentTask),
                    Integer.toString(taskId2CorrectAnswer.get(currentTask)),
                    Double.toString(correctVotes / totalVotes)});


            totalResultsRows.addAll(crossValidatedRows);
            writerResults.writeRowsAndClose(crossValidatedRows.toArray(new String[crossValidatedRows.size()][4]));
        }

        Curve.PrimitivesBuilder builder = new Curve.PrimitivesBuilder();
        int count = 0;
        for (int i = 0; i < positivePredicitions.length; i++) {
            if (positivePredicitions[i] < 1) {
                count++;
            }
        }
        builder.actuals(truth);
        builder.predicteds(positivePredicitions);
        Curve curve = builder.build();
        double rocArea = curve.rocArea();
        double prArea = curve.prArea();
        System.out.println("rocArea = " + rocArea);
        System.out.println("prArea = " + prArea);
        double[][] rocPoints = curve.rocPoints();
        double[][] prPoints = curve.prPoints();

        double[] rocPointsX = new double[rocPoints.length];
        double[] rocPointsY = new double[rocPoints.length];
        double[] prPointsX = new double[prPoints.length];
        double[] prPointsY = new double[prPoints.length];

        for (int i = 0; i < rocPoints.length; i++) {
            rocPointsX[i] = rocPoints[i][0];
            rocPointsY[i] = rocPoints[i][1];
            prPointsX[i] = prPoints[i][0];
            prPointsY[i] = prPoints[i][1];
        }

        XYChart chart = QuickChart.getChart("ROC Curve", "False Positive Rate", "True Positive Rate", "ROC",
                rocPointsX, rocPointsY);
        XYChart chart2 = QuickChart.getChart("PR Curve", "Recall", "Precision", "PR", prPointsX, prPointsY);

        new SwingWrapper(chart).displayChart();
        new SwingWrapper(chart2).displayChart();

        List<Double> predictions = new ArrayList<>();
        for (int i = 0; i < positivePredicitions.length; i++) {
            predictions.add(positivePredicitions[i]);
        }
        double logLoss = 0;
        predictions = predictions.stream().sorted().collect(Collectors.toList());

        int iMax = 0;
        double fMeasureMax = 0;
        double precisionBest = 0;
        double recallBest = 0;
        double[] threshold = new double[truth.length - 1];
        double[] measure = new double[truth.length - 1];
        for (int i = 0; i <= truth.length; i++) {
            double precision = curve.precision(i);
            double recall = curve.recall(i);
            double fMeasure = 2 * (precision * recall) / (precision + recall);
            if (fMeasure > fMeasureMax) {
                iMax = i;
                fMeasureMax = fMeasure;
                precisionBest = precision;
                recallBest = recall;
            }
            if (i < threshold.length - 1) {
                threshold[i] = predictions.get(predictions.size() - 1 - i);
                measure[i] = fMeasure;
            }
        }
        System.out.println("precision best = " + precisionBest);
        System.out.println("recall best = " + recallBest);
        System.out.println("f1 measure best = " + fMeasureMax);
        XYChart chart3 = QuickChart.getChart("Threshold", "Threshold", "F1", "Threshold", threshold, measure);
        new SwingWrapper(chart3).displayChart();

        int[] confusionMatrix = curve.confusionMatrix(iMax);
        System.out.println("confusion matrix = {" + confusionMatrix[0] + ", " + confusionMatrix[1] + ", " + confusionMatrix[2] + ", " + confusionMatrix[3] + "}");
        for (int i = 0; i < predictions.size(); i++) {
            if (predictions.size() - i > iMax) {
                logLoss += Math.log(1 - predictions.get(i));
            } else {
                logLoss += Math.log(predictions.get(i));
            }
        }
        logLoss = -((double) 1 / (double) predictions.size()) * logLoss;
        System.out.println("log loss = " + logLoss);
        List<Double> scores = new ArrayList<>();
        userId2Score.keySet().forEach(userId -> {
            scores.add(userId2Score.get(userId).stream().mapToDouble(x -> x).average().getAsDouble());
        });

        Histograms.drawHistogram(scores, "users behavioral score", 30, 2);

        TsvWriter majorityWeightedWriter = new TsvWriter(new PrintWriter(new FileWriter("majority_weighted_aggregated"
                + ".tsv")), new TsvWriterSettings());
        majorityWeightedWriter.writeHeaders("task_id", "answer", "confidence");
        double majorityVoteWeightedQuality =
                majorityWeightedRows.stream().mapToDouble(s -> Double.parseDouble(s[2])).average().getAsDouble();
        majorityWeightedWriter.writeRowsAndClose(majorityWeightedRows.toArray(new String[majorityWeightedRows.size()][3]));
        System.out.println("majority weighted vote quality = " + majorityVoteWeightedQuality);

        writerTotalResults.writeRowsAndClose(totalResultsRows.toArray(new String[totalResultsRows.size()][4]));
        TsvWriter usersScoreWriters = new TsvWriter(new PrintWriter(new FileWriter("users_score" + ".tsv")),
                new TsvWriterSettings());
        usersScoreWriters.writeHeaders("userId", "score", "correct part of questions", "total number of questions");
        ArrayList<String[]> usersScores = new ArrayList<>();
        userId2Score.keySet().forEach(userId -> usersScores.add(new String[]{Integer.toString(userId),
                Double.toString(userId2Score.get(userId).stream().mapToDouble(x -> x).average().getAsDouble()),
                Double.toString(userId2CorrectPart.get(userId).stream().mapToDouble(x -> x).sum() / userId2CorrectPart.get(userId).size()), Integer.toString(userId2CorrectPart.get(userId).size())}));
        usersScoreWriters.writeRowsAndClose(usersScores.toArray(new String[usersScores.size()][4]));
    }

    static void createList(ArrayList<List<Double>> listToInsert, List<ObjectDescription> objectDescriptions) {
        for (int k = 0; k < objectDescriptions.size(); k++) {
            ObjectDescription objectDescription = objectDescriptions.get(k);
            listToInsert.add(Arrays.asList(objectDescription.answerChanges, objectDescription.timeForAnswerApproval,
                    objectDescription.idlenessTime, objectDescription.timeFromStartOfSession,
                    objectDescription.meanTimeForTask, objectDescription.meanTimeForUser,
                    objectDescription.answerVariants, objectDescription.answerTime,
                    objectDescription.dawidSkeneErrorRate));
        }
    }

    static void normalize(List<ObjectDescription> descriptions, ToDoubleFunction<ObjectDescription> getFeature,
                          ObjDoubleConsumer<ObjectDescription> setFeature) {
        double max = descriptions.stream().mapToDouble(getFeature).max().getAsDouble();
        double min = descriptions.stream().mapToDouble(getFeature).min().getAsDouble();
        double divisor = max - min;
        descriptions.stream().forEach(description -> setFeature.accept(description,
                (getFeature.applyAsDouble(description) - min) / divisor));
    }
}
