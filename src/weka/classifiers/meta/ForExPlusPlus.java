/*  Implementation of ForEx++ - "A New Framework for Knowledge Discovery 
    from Decision Forests" by Md Nasim Adnan and Md Zahidul Islam. 
    Copyright (C) <2019>  <Michael Furner>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>. 
    
    Author contact details: 
    Name: Michael Furner
    Email: mfurner@csu.edu.au
    Location: 	School of Computing and Mathematics, Charles Sturt University,
    			Bathurst, NSW, Australia, 2795.
 */
package weka.classifiers.meta;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Vector;

//import weka.associations.gsp.Element;
import weka.classifiers.SingleClassifierEnhancer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import java.util.regex.*;  
import weka.classifiers.trees.J48;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;

/**
 * <!-- globalinfo-start -->
 * Implementation of the knowledge discovery framework ForEx++, which was
 * published in:<br>
 * <br>adult.arff
 * Md Nasim Adnan and Md Zahidul Islam: ForEx++: A New Framework for Knowledge
 * Discovery from Decision Forests In: Australasian Journal of Information 
 * Systems Vol 21, 2017.<br>
 * <br>
 * This algorithm processes a SysFor forest and provides a list of high-quality
 * rules that account for each class.
 * <!-- globalinfo-end -->
 *
 *
 * @author Michael Furner
 * @version $Revision: 1.0$
 */
public class ForExPlusPlus extends SingleClassifierEnhancer {

    /**
     * For serialization.
     */
    private static final long serialVersionUID = -589120800957072995L;

    /**
     * The classifier, which must be a SysFor. 
     */
//    protected Classifier m_Classifier = new SysFor();

    /**
     * All of the rules extracted from the Decision Forest
     */
    protected RuleCollection extractedRules = null;
    
    /**
     * The final high quality rules output by ForEx++
     */
    protected RuleCollection finalRules = null;
    
    private boolean printClassifier = false;
    
    private boolean removeZeroCoverageRules = true;
    
    private boolean groupRulesViaClassValue = true;
    
    public static final int SORT_ACCURACY = 1;
    public static final int SORT_COVERAGE = 2;
    public static final int SORT_LENGTH = 3;
    
    public static final Tag[] TAGS_SORT = {
        new Tag(SORT_ACCURACY, "Sort by rule accuracy."),
        new Tag(SORT_COVERAGE, "Sort by rule coverage."),
        new Tag(SORT_LENGTH, "Sort by rule length.")
    };
    
    protected int sortType = SORT_ACCURACY;

    /**
    * Default constructor.
    */
    public ForExPlusPlus() {
        m_Classifier = new weka.classifiers.trees.SysFor();
    }

    /**
     * String describing default classifier.
     * 
     * @return the default classifier classname
     */
    @Override
    protected String defaultClassifierString() {
        return "weka.classifiers.trees.SysFor";
    }

    
    /**
     * Set the options for ForEx++. The only options are the SysFor parameters.
     * @param options
     * @throws Exception
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        printClassifier = Utils.getFlag("P", options);
        removeZeroCoverageRules = Utils.getFlag("Z", options);
        
        String sortString = Utils.getOption('E', options);
        if(sortString.length() != 0) {
            if(sortString.equals("acc")) {
                setSortType(new SelectedTag(SORT_ACCURACY, TAGS_SORT));
            }
            else if(sortString.equals("cov")) {
                setSortType(new SelectedTag(SORT_COVERAGE, TAGS_SORT));
            }
            else if(sortString.equals("len")) {
                setSortType(new SelectedTag(SORT_LENGTH, TAGS_SORT));
            }
            else {
                throw new IllegalArgumentException("Invalid sort method.");
            }
        }
        
        super.setOptions(options);
    }

    /**
     * Gets the current settings of the classifier.
     *
     * @return the current setting of the classifier
     */
    @Override
    public String[] getOptions() {
        Vector<String> result = new Vector<String>();
        
        if(printClassifier)
            result.add("-P");
        
        if(removeZeroCoverageRules)
            result.add("-Z");
        
        result.add("-E");
        switch(sortType) {
            case SORT_ACCURACY:
                result.add("acc");
                break;
            case SORT_COVERAGE:
                result.add("cov");
                break;
            case SORT_LENGTH:
                result.add("len");
                break;
        }

        Collections.addAll(result, super.getOptions());

        return result.toArray(new String[result.size()]);

    }

    /**
     * Builds and parses the SysFor to get the rules as specified by the ForEx++
     * algorithm.
     *
     * @param data - data with which to build the classifier
     * @throws java.lang.Exception
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {

        getCapabilities().testWithFail(data);
        data = new Instances(data);

        //ensure the classifier is a SysFor
        if(m_Classifier.getClass().getName() != "weka.classifiers.trees.SysFor") {
            J48 useless = new J48();
            useless.buildClassifier(data);
            m_Classifier = useless;
            return;
        }
        
        
        //if this is a dataset with only the class attribute
        if (data.numAttributes() == 1) {
            J48 useless = new J48();
            useless.buildClassifier(data);
            m_Classifier = useless;
            return;
        }
        
        
        //build SysFor
        m_Classifier.buildClassifier(data);

        //get the proportion of records from each class
        int[] classCounts = data.attributeStats(data.classIndex()).nominalCounts;
        double[] classRatio = new double[classCounts.length];
        for (int i = 0; i < classCounts.length; i++) {
            classRatio[i] = (double)classCounts[i] / data.size();
        }
        
        //extract the rules as our Rule class    
        String ruleStrings = m_Classifier.toString();        
        extractedRules = extractRulesFromRuleStrings(ruleStrings, data.numInstances(), data.classAttribute());
        
        //remove useless rules
        if(removeZeroCoverageRules) {
            extractedRules = extractedRules.getRulesGEQAccuracy(Double.MIN_VALUE);
        }
        
        //collect the intersection of good rules for each class value
        RuleCollection[] intersections = new RuleCollection[data.classAttribute().numValues()];
        
        for(int i = 0; i < data.classAttribute().numValues(); i++) {
            
            RuleCollection rulesForClass = extractedRules.getRulesByClass(i);
            
            //get rules with higher than average mean
            double meanAccuracy = rulesForClass.meanAccuracy();
            RuleCollection rAcc = rulesForClass.getRulesGEQAccuracy(meanAccuracy);
            
            //get rules with higher than average coverage
            double meanCoverage = rulesForClass.meanCoverage();
            RuleCollection rCov = rulesForClass.getRulesGEQCoverage(meanCoverage);

            double meanRuleLength = rulesForClass.meanRuleLength();
            RuleCollection rLen = rulesForClass.getRulesLEQLength(meanRuleLength);
            
            RuleCollection rAccCovLen = rAcc.intersection(rCov).intersection(rLen);
            
            intersections[i] = rAccCovLen;
            
        }
        
        //combine into the ForEx++ result set
        finalRules = intersections[0];
        for(int i = 1; i < intersections.length; i++) {
            finalRules = finalRules.merge(intersections[i]);
        }
        

    }
    
    private int countMatches(String toSearch, String toFind) {
        
        return toSearch.length() - toSearch.replace(toFind, "").length();
        
    }
    
    private RuleCollection extractRulesFromRuleStrings(String ruleStrings, int numRecords, Attribute classAttr) {
        
        HashSet<Rule> ruleVec = new HashSet<Rule>();
        String[] rules = ruleStrings.split("\n");
                
        int numberOfLeaves = countMatches(ruleStrings, "(");
        String[] leaves = new String[numberOfLeaves];
        int leafIndex = 0;

        for(int j = 0; j < rules.length; j++) {

            String rule = rules[j];

            if(rule.contains("(")) { //if we have a leaf
                int numberOfPipes = countMatches(rule, "|");

                //if we're at root, save rule and quit
                if(numberOfPipes == 0) {
                    leaves[leafIndex] = rule;
                    leafIndex++;
                    continue;
                }

                //we want to go up the tree and find the split points
                for(int k = j - 1; k >= 0; k--) {

                    String testerRule = rules[k];
                    int testerRulePipes = countMatches(testerRule, "|");

                    //continue if this is on the same level
                    if(testerRulePipes == numberOfPipes) 
                        continue;

                    //we've found the parent split
                    if(testerRulePipes < numberOfPipes ) {
                        numberOfPipes = testerRulePipes;
                        rule = testerRule + rule;

                        //if we're at root, save rule and quit
                        if(testerRulePipes == 0) {
                            leaves[leafIndex] = rule.replaceAll("(\\|   )+", " && ");
                            leafIndex++;
                            break;
                        }
                    } //end if parent

                } //end split point finder

            }

        }
        
        //we have string representations of all the rules, now to extract their information
        String ruleRegex = "^(.+): ([a-zA-Z0-9-_!@#$%^*~'\"\\&]+)(| \\{[a-zA-Z0-9;,-_!@#$%^*~'\"\\&]+\\}) \\(([0-9.]+)(|/[0-9.]+)\\)";
        Pattern regex = Pattern.compile(ruleRegex);
        for(int i = 0; i < numberOfLeaves; i++) {
            if(leaves[i] != null) {

                Matcher matcher = regex.matcher(leaves[i]);
                matcher.matches();
                try {
                    String ruleText = matcher.group(1);
                    String classPredictedText = matcher.group(2);
                    String distributionText = matcher.group(3);
                    String recordsInLeafText = matcher.group(4);
                    String misclassifiedText = matcher.group(5);

                    //if there's no text there, then we have 100% accuracy in rule
                    double misclassified = 0;
                    if(!"".equals(misclassifiedText)) {
                        misclassifiedText = misclassifiedText.replace("/", "");
                        misclassified = Double.parseDouble(misclassifiedText);
                    }

                    //accuracy is calculated by (support - misclassification) / number of rules in leaf
                    double recordsInLeaf = Double.parseDouble(recordsInLeafText);
                    double accuracy = 0;
                    if (recordsInLeaf != 0) {
                        accuracy = (recordsInLeaf - misclassified) / (recordsInLeaf);
                    }

                    //support is expressed as a fraction of the whole dataset
                    double support = recordsInLeaf / numRecords;

                    //rule length can be grabbed by counting number of &&s
                    int ruleLength = ruleText.split("&&").length + 1;

                    //identify the class index
                    int predictedClass = classAttr.indexOfValue(classPredictedText);

                    //class distribution formatting
                    double[] classDistribution = null;
                    if(!"".equals(distributionText)) {

                        classDistribution = new double[classAttr.numValues()];

                        distributionText = distributionText.replace("{", "");
                        distributionText = distributionText.replace("}", "");
                        distributionText = distributionText.replace(" ", "");

                        String[] dist = distributionText.split(";");
                        for (int k = 0; k < dist.length; k++) {

                            String val = dist[k];
                            String[] distVals = val.split(",");
                            classDistribution[k] = Double.parseDouble(distVals[1]);

                        }
                    }

                    //set up the rule and add it to the vector
                    Rule theRule = new Rule(ruleText, accuracy, support, ruleLength,
                            predictedClass, classPredictedText, classDistribution, sortType);
                    ruleVec.add(theRule);
                }
                catch(Exception e) {
                    System.out.println(leaves[i]);
                }
            }
        }
        
        
        return new RuleCollection(ruleVec);
        
    }

    /**
     * Passes the classification through to the built SysFor
     *
     * @param instance - the instance to be classified
     * @return probablity distribution for this instance's classification
     * @throws java.lang.Exception
     */
    @Override
    public double[] distributionForInstance(Instance instance) throws java.lang.Exception {

        return m_Classifier.distributionForInstance(instance);

    }

    /**
     * Main method for testing this class.
     *
     * @param argv should contain the following arguments: -t training file [-T
     * test file] [-c class index]
     */
    public static void main(String[] argv) throws Exception {
        runClassifier(new ForExPlusPlus(), argv);
    }

    /**
     * Returns capabilities of algorithm
     *
     * @return Weka capabilities of ForExPlusPlus
     */
    @Override
    public Capabilities getCapabilities() {

        Capabilities result = super.getCapabilities();   // returns the object from weka.classifiers.Classifier

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);
        result.disable(Capabilities.Capability.RELATIONAL_ATTRIBUTES);
        result.disable(Capabilities.Capability.STRING_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.disable(Capabilities.Capability.NUMERIC_CLASS);
        result.disable(Capabilities.Capability.DATE_CLASS);
        result.disable(Capabilities.Capability.RELATIONAL_CLASS);
        result.disable(Capabilities.Capability.UNARY_CLASS);
        result.disable(Capabilities.Capability.NO_CLASS);
        result.disable(Capabilities.Capability.STRING_CLASS);
        return result;

    }

    /**
     * Return a description suitable for displaying in the
     * explorer/experimenter.
     *
     * @return a description suitable for displaying in the
     * explorer/experimenter
     */
    public String globalInfo() {
        return "For more information, see:\n\n" + getTechnicalInformation().toString();
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "Md Nasim Adnan & Md Zahidul Islam");
        result.setValue(Field.YEAR, "2017");
        result.setValue(Field.TITLE, "ForEx++: A New Framework for Knowledge Discovery from Decision Forests");
        result.setValue(Field.JOURNAL, "Australasian Journal of Information Systems");
        result.setValue(Field.PUBLISHER, "ACS");
        result.setValue(Field.VOLUME, "21");
        result.setValue(Field.URL, "https://doi.org/10.3127/ajis.v21i0.1539");

        return result;

    }

    /**
     * String output of ForEx++ algorithm.
     * @return final rules as string
     */
    @Override
    public String toString() {

        if (finalRules != null) {
             
            StringBuilder out = new StringBuilder("");
            out.append(finalRules.toString(groupRulesViaClassValue));
            
            if(printClassifier) {
                out.append("\n").append(m_Classifier.toString());
            }
        
            return out.toString();
        }
        else 
            return "ForEx++ not built!\nWeka ForEx++ can currently only parse SysFor.";

    }

    /**
     * List the possible options from the superclass
     * @return Options Enumerated
     */
    @Override
    public Enumeration<Option> listOptions() {

        Vector<Option> newVector = new Vector<Option>();
        newVector.addElement(new Option("\tWhether to print the SysFor that the "
                + "ForEx++ rules were selected from.\n"
                + "\t(default false)", "P", 0, "-P"));
        newVector.addElement(new Option("\tWhether to remove rules with no coverage"
                + "before calculating mean coverage, support, and rule length.\n"
                + "\t(default false)", "P", 0, "-P"));
        newVector.addElement(new Option("\tSort Method for Displaying Rules.\n"
                + "\t(Default = sort by rule accuracy)",
                  "E", 1, "-E <acc | cov | len>"));

        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    }
    
    private class Rule implements Serializable, Comparable {

        private static final long serialVersionUID = -5891208000957072995L;

        private final String ruleText;
        private final double accuracy;
        private final double coverage;
        private final int length;
        private final int predictedClass;
        private final String predictedClassLabel;
        private final double[] classDistribution;
        private final int sortMethod;
        
        public Rule(String ruleText, double accuracy, double coverage, int length,
                    int predictedClass, String predictedClassLabel, double[] classDistribution,
                    int sortMethod) {
            this.ruleText = ruleText;
            this.accuracy = accuracy;
            this.coverage = coverage;
            this.length = length;
            this.predictedClass = predictedClass;
            this.predictedClassLabel = predictedClassLabel;
            this.classDistribution = classDistribution;
            this.sortMethod = sortMethod;
        }
        
        public String getRuleText() {
            return ruleText;
        }
        
        public double getAccuracy() {
            return accuracy;
        }
        
        public double getCoverage() {
            return coverage;
        }
        
        public int getLength() {
            return length;
        }
        
        public int getPredictedClassIndex() {
            return predictedClass;
        }
        
        public String getPredictedClassLabel() {
            return predictedClassLabel;
        }
        
        public double[] getClassDistribution() {
            return classDistribution;
        }
        
        @Override
        public String toString() {
            StringBuilder outString = new StringBuilder(ruleText).append(": ")
                    .append(predictedClassLabel);
            
            outString.append(". Confidence: ").append(String.format("%.3f", accuracy))
                    .append("; Coverage: ").append(String.format("%.3f", coverage))
                    .append(";");
            
            return outString.toString();
        }

        @Override
        public int compareTo(Object o) {
            
            Rule other = (Rule)o;
            
            switch(sortMethod) {
            case SORT_ACCURACY:
                if(accuracy > other.getAccuracy())
                    return 1;
                else if(accuracy < other.getAccuracy())
                    return -1;
                else
                    return 0;
            case SORT_COVERAGE:
                if(coverage > other.getAccuracy())
                    return 1;
                else if(coverage < other.getAccuracy())
                    return -1;
                else
                    return 0;
            case SORT_LENGTH:
                if(length > other.getAccuracy())
                    return 1;
                else if(length < other.getAccuracy())
                    return -1;
                else
                    return 0;
            }
            
            return 0;
            
        }
        
    }
    
     private class RuleCollection implements Serializable {
         
        private static final long serialVersionUID = -5891208009570723995L;

         
        private final HashSet<Rule> rules;

        public RuleCollection(HashSet<Rule> rules) {
            this.rules = rules;
        }
        
        public HashSet<Rule> getRules() {
            return rules;
        }
        
        public double meanAccuracy() {
            
            double mean = 0;
            
                        
            for(Rule r : rules) {
                mean += r.getAccuracy();
            }
            
            mean /= rules.size();
            
            return mean;
            
        }
        
        public double meanCoverage() {
            
            double mean = 0;
            
            for(Rule r : rules) {
                mean += r.getCoverage();
            }
            
            mean /= rules.size();
            
            return mean;
            
        }
        
        public double meanRuleLength() {
            
            double mean = 0;
            
            for(Rule r : rules) {
                mean += r.getLength();
            }
            
            mean /= rules.size();
            
            return mean;
            
        }
        
        public RuleCollection getRulesGEQAccuracy(double accuracy) {
            
            HashSet<Rule> newRules = new HashSet<Rule>();
            
            for(Rule r : rules) {
                if(r.getAccuracy() >= accuracy) {
                    newRules.add(r);
                }
            }
            
            return new RuleCollection(newRules);
        }
        
        public RuleCollection getRulesGEQCoverage(double coverage) {
            
            HashSet<Rule> newRules = new HashSet<Rule>();
            
            for(Rule r : rules) {
                if(r.getCoverage()>= coverage) {
                    newRules.add(r);
                }
            }
            
            return new RuleCollection(newRules);
        }
        
        public RuleCollection getRulesLEQLength(double length) {
            
            HashSet<Rule> newRules = new HashSet<Rule>();
            
            for(Rule r : rules) {
                if(r.getLength() <= length) {
                    newRules.add(r);
                }
            }
            
            return new RuleCollection(newRules);
        }
        
        public RuleCollection merge(RuleCollection otherRules) {
            
            HashSet<Rule> newRuleSet = new HashSet<Rule>();
            newRuleSet.addAll(rules);
            newRuleSet.addAll(otherRules.getRules());
            
            RuleCollection newRules = new RuleCollection(newRuleSet);
            
            return newRules;
            
        }
        
        public RuleCollection getRulesByClass(int classIndex) {
            HashSet<Rule> newRules = new HashSet<Rule>();
            
            for(Rule r : rules) {
                if(r.getPredictedClassIndex() == classIndex) {
                    newRules.add(r);
                }
            }
            
            return new RuleCollection(newRules);
        }
        
        public RuleCollection intersection(RuleCollection otherRules) {
            
            HashSet<Rule> newRuleSet = new HashSet<Rule>(rules);
            newRuleSet.retainAll(otherRules.getRules());
            
            return new RuleCollection(newRuleSet);
            
        }
        
        public String toString(boolean group) {
            
            StringBuilder out = new StringBuilder("ForEx++ Rules:\n");
            
            if(!group) {
                
                ArrayList<Rule> ruleList = new ArrayList<>();
                for(Rule r : rules) {
                    ruleList.add(r);
                }
                
                Collections.sort(ruleList, Collections.reverseOrder());
                
                for(Rule r : ruleList) {
                    out.append(r.toString()).append("\n");
                }
            }
            else { //group by class values
                
                HashMap<String, ArrayList<Rule>> classMap = new HashMap<>();
                
                //add all the rules to a corresponding class vector
                for(Rule r : rules) {
                    
                    if(!classMap.containsKey(r.getPredictedClassLabel())) {
                        classMap.put(r.getPredictedClassLabel(), new ArrayList<>());
                    }
                    
                    classMap.get(r.getPredictedClassLabel()).add(r);
                    
                } 
                
                //iterate over these classes individually and add to the output
                for(String k : classMap.keySet()) {
                    
                    out.append("Rules for class value ").append(k).append(": \n");
                    
                    Collections.sort(classMap.get(k), Collections.reverseOrder());
                    
                    for (Rule rule : classMap.get(k)) {
                        
                        out.append(rule.toString()).append("\n");
                        
                    }
                    
                    out.append("\n\n");
                    
                }
                
            } //end grouping
            
            return out.toString();
        }
        
        @Override
        public String toString() {
            return toString(false);
        }
        
         
       
    }
     
    public boolean isPrintClassifier() {
        return printClassifier;
    }

    public void setPrintClassifier(boolean printClassifier) {
        this.printClassifier = printClassifier;
    }
    
    public String printClassifierTipText() {
        return "Whether to print the SysFor that the ForEx++ rules were selected from.";
    }
    
    public boolean isRemoveZeroCoverageRules() {
        return removeZeroCoverageRules;
    }

    public void setRemoveZeroCoverageRules(boolean removeZeroCoverageRules) {
        this.removeZeroCoverageRules = removeZeroCoverageRules;
    }

    public String removeZeroCoverageRulesTipText() {
        return "Whether to remove rules with no coverage before calculating mean coverage, support, and rule length.";
    }
    
    public boolean isGroupRulesViaClassValue() {
        return groupRulesViaClassValue;
    }

    public void setGroupRulesViaClassValue(boolean groupRulesViaClassValue) {
        this.groupRulesViaClassValue = groupRulesViaClassValue;
    }

    public String groupRulesViaClassValueTipText() {
        return "Whether to group rules via their class values.";
    }
    
    public SelectedTag getSortType() {
        return new SelectedTag(sortType, TAGS_SORT);
    }
    
    public void setSortType(SelectedTag newSortType) {
        if(newSortType.getTags() == TAGS_SORT) {
            sortType = newSortType.getSelectedTag().getID();
        }
    }
    
    public String sortTypeTipText() {
        return "Method to sort the rules when displayed.";
    }

}
