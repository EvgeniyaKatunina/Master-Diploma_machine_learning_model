import org.knowm.xchart.*;

import java.awt.*;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Histograms {
    static void drawHistogram(List<Double> dataValues, String name, int numBins, int scale) {
        double min = dataValues.stream().mapToDouble(it -> it).min().orElse(Double.NaN);
        double max = dataValues.stream().mapToDouble(it -> it).max().orElse(Double.NaN);
        drawHistogram(dataValues, name, numBins, scale, min, max);
    }

    static void drawHistogram(List<Double> dataValues, String name, int numBins, int scale, double min, double max) {
        List<Double> keys = calcHistogramKeys(min, max, numBins, scale);

        Histogram histogram = new Histogram(dataValues, numBins, min, max);

        CategoryChart chart = new CategoryChartBuilder().width(1200).height(800)
                .title(name)
                .yAxisTitle("Frequency")
                .build();

        chart.getStyler().setAvailableSpaceFill(0.99);
        chart.getStyler().setLegendVisible(false);
        chart.getStyler().setOverlapped(true);
        chart.getStyler().setStacked(true);
        chart.getStyler().setAxisTickLabelsFont(new Font("Arial", Font.PLAIN, 13));

        chart.addSeries("data", keys, histogram.getyAxisData());

        try {
            BitmapEncoder.saveBitmap(chart, name, BitmapEncoder.BitmapFormat.PNG);
        } catch (IOException e) {
            e.printStackTrace();
        }

        new SwingWrapper<>(chart).displayChart();
    }

    private static List<Double> calcHistogramKeys(double min, double max, int numBins, int scale) {
        final double binSize = (max - min) / numBins;
        return IntStream.range(0, numBins).mapToObj(it -> round(it * binSize, scale)).collect(Collectors.toList());
    }

    private static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        BigDecimal bd = new BigDecimal(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }
}
