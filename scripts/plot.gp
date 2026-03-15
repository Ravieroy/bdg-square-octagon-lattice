#!/usr/bin/env gnuplot

### ------------------------------
### Argument handling
### ------------------------------
if (ARGC < 1) {
    print "Usage:"
    print "  gnuplot -c plot.gp datafile.dat"
    print "  gnuplot -c plot.gp datafile.dat all"
    print "  gnuplot -c plot.gp datafile.dat cols \"2 3 4\""
    exit
}

datafile = ARG1
plot_all = (ARGC >= 2 && ARG2 eq "all")
use_cols = (ARGC >= 3 && ARG2 eq "cols")

outfile = "plot.png"

### ------------------------------
### Terminal & output
### ------------------------------
set terminal pngcairo size 900,600 enhanced font "Helvetica,14"
set output outfile

set encoding utf8
set datafile separator whitespace

### ------------------------------
### Aesthetic settings
### ------------------------------
set border lw 1.5
set tics out scale 0.75
set grid back lw 0.5 lc rgb "#bbbbbb"
set key top left box opaque

set style line 1 lc rgb "#1f77b4" lw 2 pt 7 ps 1.2

### ------------------------------
### Labels & title
### ------------------------------
set xlabel "x"
set ylabel "y"
set title sprintf("Plot of %s", datafile)

### ------------------------------
### Use header row for legend
### ------------------------------
set key autotitle columnhead

### ------------------------------
### Detect number of columns
### ------------------------------
stats datafile nooutput
ncols = STATS_columns

### ------------------------------
### Plot logic
### ------------------------------
if (use_cols) {

    # User-defined column list, e.g. "2 3 4"
    collist = ARG3

    plot for [i=1:words(collist)] \
        datafile using 1:int(word(collist,i)) with linespoints

} else if (!plot_all || ncols == 2) {

    # Default: column 1 vs column 2
    plot datafile using 1:2 with linespoints ls 1

} else {

    # Column 1 vs all remaining columns
    plot for [col=2:ncols] \
        datafile using 1:col with linespoints
}

unset output
print sprintf("Saved plot to %s", outfile)

