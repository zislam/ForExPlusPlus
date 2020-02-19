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
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Vector;
import java.util.Arrays;

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
import weka.classifiers.trees.RandomForest;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;

/**
 * <!-- globalinfo-start -->
 * Implementation of the knowledge discovery framework ForEx++, which was
 * published in:<br>
 * <br>
 * Md Nasim Adnan and Md Zahidul Islam: ForEx++: A New Framework for Knowledge
 * Discovery from Decision Forests In: Australasian Journal of Information 
 * Systems Vol 21, 2017.<br>
 * <br>
 * This algorithm processes a decision forest and provides a list of high-quality
 * rules that account for each class.
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -P
 *  Whether to print the decision forest that the ForEx++ rules were selected from
 *  (default false)
 * </pre>
 * 
 * <pre>
 * -Z
 *  Whether to remove rules with no coverage before calculating mean coverage, 
 *  support, and rule length
 *  (default true)
 * </pre>
 * 
 * <pre>
 * -GC
 *  Whether to group rules by class value in the final output.
 *  (default true)
 * </pre>
 * 
 * <pre>
 * -E &lt;acc | cov | len&gt;
 *  Sort Method for Displaying Rules.
 *  (Default = sort by rule accuracy)
 * </pre>
 * 
 * <pre>
 * -UA
 *  Whether to use accuracy in selecting ForEx++ rules
 *  (default true)
 * </pre>
 *
 * <pre>
 * -UC
 *  Whether to use coverage in selecting ForEx++ rules
 *  (default true)
 * </pre>
 * 
 * <pre>
 * -UR
 *  Whether to use rule length in selecting ForEx++ rules
 *  (default true)
 * </pre>
 *
 * <!-- options-end -->
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
     * All of the rules extracted from the Decision Forest
     */
    protected RuleCollection extractedRules = null;
    
    /**
     * The final high quality rules output by ForEx++
     */
    protected RuleCollection finalRules = null;
    
    /** Whether to print the classifier as well as ForEx++ rules */
    private boolean printClassifier = false;
    
    /** Whether to remove rules with no coverage before calculating mean coverage, support, and rule length */
    private boolean removeZeroCoverageRules = true;
    
    /** Whether to group rules by class value in the final output. */
    private boolean groupRulesViaClassValue = true;
    
    /** Whether to use accuracy in selecting ForEx++ rules */
    private boolean useAccuracy = true;
    
    /** Whether to use coverage in selecting ForEx++ rules */
    private boolean useCoverage = true;
    
    /** Whether to use rule length in selecting ForEx++ rules */
    private boolean useRuleLength = true;
    
    /** Store how many rules are actually found by the decision forest before we extract the ForEx++ rules. */
    private int totalRulesFromClassifier = 0;
        
    /** Sort type: accuracy (highest accuracy first). */
    public static final int SORT_ACCURACY = 1;
    /** Sort type: coverage (highest coverage first). */
    public static final int SORT_COVERAGE = 2;
    /** Sort type: rule length (shortest first). */
    public static final int SORT_LENGTH = 3;
    
    /** Tags for displaying sort types in the GUI. */
    public static final Tag[] TAGS_SORT = {
        new Tag(SORT_ACCURACY, "Sort by rule accuracy (highest accuracy first)."),
        new Tag(SORT_COVERAGE, "Sort by rule coverage (highest coverage first)."),
        new Tag(SORT_LENGTH, "Sort by rule length (shortest first).")
    };
    
    /** Sort Method for Displaying Rules */
    protected int sortType = SORT_ACCURACY;
    
    /** Enum for holding different build statuses */
    enum BuildStatus {
        BS_BUILT, BS_UNBUILT, BS_NOTCOMPATIBLE, BS_RANDOMFOREST_PRINT, 
        BS_NOUSEFLAGS, BS_ONEATTRIBUTE
    };
    
    /** The build status of this ForEx++ object */
    private BuildStatus buildStatus = BuildStatus.BS_UNBUILT;

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
     * Parse the options for ForEx++.
     * 
     * * <!-- options-start --> Valid options are:
     * <p/>
     * 
     * <pre>
     * -P
     *  Whether to print the decision forest that the ForEx++ rules were selected from
     *  (default false)
     * </pre>
     * 
     * <pre>
     * -Z
     *  Whether to remove rules with no coverage before calculating mean coverage, 
     *  support, and rule length
     *  (default true)
     * </pre>
     * 
     * <pre>
     * -GC
     *  Whether to group rules by class value in the final output.
     *  (default true)
     * </pre>
     * 
     * <pre>
     * -E &lt;acc | cov | len&gt;
     *  Sort Method for Displaying Rules.
     *  (Default = sort by rule accuracy)
     * </pre>
     * 
     * <pre>
     * -UA
     *  Whether to use accuracy in selecting ForEx++ rules
     *  (default true)
     * </pre>
     *
     * <pre>
     * -UC
     *  Whether to use coverage in selecting ForEx++ rules
     *  (default true)
     * </pre>
     * 
     * <pre>
     * -UR
     *  Whether to use rule length in selecting ForEx++ rules
     *  (default true)
     * </pre>
     *
     * <!-- options-end -->
     * 
     * @param options
     * @throws Exception
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        printClassifier = Utils.getFlag("P", options);
        removeZeroCoverageRules = Utils.getFlag("Z", options);
        groupRulesViaClassValue = Utils.getFlag("GC", options);
        
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
        
        useAccuracy = Utils.getFlag("UA", options);
        useCoverage = Utils.getFlag("UC", options);
        useRuleLength = Utils.getFlag("UR", options);
        
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
        
        if(groupRulesViaClassValue)
            result.add("-GC");
        
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
        
        if(useAccuracy) {
            result.add("-UA");
        }
        if(useCoverage) {
            result.add("-UC");
        }
        if(useRuleLength) {
            result.add("-UR");
        }

        Collections.addAll(result, super.getOptions());

        return result.toArray(new String[result.size()]);

    }

    /**
     * Builds and parses the decision forest to get the rules as specified by the ForEx++
     * algorithm.
     *
     * @param data - data with which to build the classifier
     * @throws java.lang.Exception
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {

        getCapabilities().testWithFail(data);
        data = new Instances(data);

        String classifierName = m_Classifier.getClass().getName();
        
        //ensure the classifier is a supported decision forest
        if(classifierName != "weka.classifiers.trees.SysFor" &&
           classifierName != "weka.classifiers.trees.ForestPA" &&
           classifierName != "weka.classifiers.trees.RandomForest") {
            buildStatus = BuildStatus.BS_NOTCOMPATIBLE;
        }
        if(data.numAttributes() == 1) { //dataset with only one attribute
            buildStatus = BuildStatus.BS_ONEATTRIBUTE;
        }
        if(!useAccuracy && !useCoverage && !useRuleLength) { //invalid options 
            buildStatus = BuildStatus.BS_NOUSEFLAGS;
 
        }
        
        //ensure that if the classifier is RandomForest, the print flag is set
        if(classifierName == "weka.classifiers.trees.RandomForest") {
            RandomForest rf = (RandomForest)m_Classifier;
            if(!rf.getPrintClassifiers()) {
                buildStatus = BuildStatus.BS_RANDOMFOREST_PRINT;
            }
        }
        
        if(buildStatus != BuildStatus.BS_UNBUILT) {     
            J48 useless = new J48();
            useless.buildClassifier(data);
            m_Classifier = useless;
            return;
        }
        
        //build decision forest
        m_Classifier.buildClassifier(data);

        //get the proportion of records from each class
        int[] classCounts = data.attributeStats(data.classIndex()).nominalCounts;
        double[] classRatio = new double[classCounts.length];
        for (int i = 0; i < classCounts.length; i++) {
            classRatio[i] = (double)classCounts[i] / data.size();
        }
        
        //extract the rules as our Rule class    
        String ruleStrings = m_Classifier.toString();        
        extractedRules = extractRulesFromRuleStrings(classifierName, ruleStrings, data.numInstances(), data.classAttribute());
        totalRulesFromClassifier = extractedRules.getRules().size();
        
        //remove useless rules
        if(removeZeroCoverageRules) {
            extractedRules = extractedRules.getRulesGEQAccuracy(Double.MIN_VALUE);
        }
        
        //collect the intersection of good rules for each class value
        RuleCollection[] intersections = new RuleCollection[data.classAttribute().numValues()];
        
        for(int i = 0; i < data.classAttribute().numValues(); i++) {
            
            RuleCollection rulesForClass = extractedRules.getRulesByClass(i);
            RuleCollection intersection = rulesForClass;
            
            if(useAccuracy) {
                //get rules with higher than average mean
                double meanAccuracy = rulesForClass.meanAccuracy();
                RuleCollection rAcc = rulesForClass.getRulesGEQAccuracy(meanAccuracy);
                intersection = intersection.intersection(rAcc);
            }
            
            if(useCoverage) {
                //get rules with higher than average coverage
                double meanCoverage = rulesForClass.meanCoverage();
                RuleCollection rCov = rulesForClass.getRulesGEQCoverage(meanCoverage);
                intersection = intersection.intersection(rCov);
            }
             
            if(useRuleLength) {
                //get rules with less than average length
                double meanRuleLength = rulesForClass.meanRuleLength();
                RuleCollection rLen = rulesForClass.getRulesLEQLength(meanRuleLength);
                intersection = intersection.intersection(rLen);
            }
                        
            intersections[i] = intersection;
            
        }
        
        //combine into the ForEx++ result set
        finalRules = intersections[0];
        for(int i = 1; i < intersections.length; i++) {
            finalRules = finalRules.merge(intersections[i]);
        }
        
        buildStatus = BuildStatus.BS_BUILT;
        

    }
    
    /**
     * Counts number of instances of substring in a string.
     *
     * @param toSearch - string to search for substring
     * @param toFind - substring to find
     * @return number of matches
     */
    private int countMatches(String toSearch, String toFind) {
        
        return toSearch.length() - toSearch.replace(toFind, "").length();
        
    }
    
    /**
     * Take the output of the classifier and turn it into a RuleCollection
     * object. This function simply redirects to the correct function for the
     * specific classifier.
     * 
     * @param className - type of classifier used
     * @param ruleStrings - output from the classifier
     * @param numRecords - number of records in the dataset
     * @param classAttr - the class attribute from the dataset
     * @return collection of rules
     */
    private RuleCollection extractRulesFromRuleStrings(String className, String ruleStrings, int numRecords, Attribute classAttr) {
        
        switch(className) {
            case "weka.classifiers.trees.SysFor":
                return extractRulesFromRuleStringsSysFor(ruleStrings, numRecords, classAttr);
            case "weka.classifiers.trees.RandomForest":
                return extractRulesFromRuleStringsRandomForest(ruleStrings, numRecords, classAttr);
            case "weka.classifiers.trees.ForestPA":
                return extractRulesFromRuleStringsForestPA(ruleStrings, numRecords, classAttr);
        }
        
        //default case that should never be run
        return new RuleCollection(new HashSet<Rule>());
        
    }
    
     /**
     * Take the output from a SysFor and turn it into a RuleCollection object.
     * 
     * @param ruleStrings - output from the classifier
     * @param numRecords - number of records in the dataset
     * @param classAttr - the class attribute from the dataset
     * @return collection of rules
     */
    private RuleCollection extractRulesFromRuleStringsSysFor(String ruleStrings, int numRecords, Attribute classAttr) {
        
        HashSet<Rule> ruleMap = new HashSet<Rule>();
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
                    int ruleLength = ruleText.split("&&").length;

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
                    Rule theRule = new Rule(ruleText, accuracy, support, recordsInLeaf, ruleLength,
                            predictedClass, classPredictedText, classDistribution, sortType);

                    ruleMap.add(theRule);
                }
                catch(Exception e) {
                    System.out.println(leaves[i]);
                }
            }
        }
        
        
        return new RuleCollection(ruleMap);
        
    }
    
    /**
     * Take the output from a ForestPA and turn it into a RuleCollection object.
     * 
     * @param ruleStrings - output from the classifier
     * @param numRecords - number of records in the dataset
     * @param classAttr - the class attribute from the dataset
     * @return collection of rules
     */
    private RuleCollection extractRulesFromRuleStringsForestPA(String ruleStrings, int numRecords, Attribute classAttr) {
        
        HashSet<Rule> ruleVec = new HashSet<Rule>();
        String[] rules = ruleStrings.split("\n");
                
        int numberOfLeaves = countMatches(ruleStrings, "/");
        String[] leaves = new String[numberOfLeaves];
        int leafIndex = 0;

        for(int j = 0; j < rules.length; j++) {

            String rule = rules[j];

            if(rule.contains("/")) { //if we have a leaf
                int numberOfPipes = countMatches(rule, "|  ");

                //if we're at root, save rule and quit
                if(numberOfPipes == 0) {
                    leaves[leafIndex] = rule;
                    leafIndex++;
                    continue;
                }

                //we want to go up the tree and find the split points
                for(int k = j - 1; k >= 0; k--) {

                    String testerRule = rules[k];
                    int testerRulePipes = countMatches(testerRule, "|  ");

                    //continue if this is on the same level
                    if(testerRulePipes == numberOfPipes) 
                        continue;

                    //we've found the parent split
                    if(testerRulePipes < numberOfPipes ) {
                        numberOfPipes = testerRulePipes;
                        rule = testerRule + rule;

                        //if we're at root, save rule and quit
                        if(testerRulePipes == 0) {
                            leaves[leafIndex] = rule.replaceAll("(\\|  )+", " && ");
                            leafIndex++;
                            break;
                        }
                    } //end if parent

                } //end split point finder

            }

        }
        
        //we have string representations of all the rules, now to extract their information
        String ruleRegex = "^(.+): ([a-zA-Z0-9-_!@#$%^*~'\"\\&]+)\\(([0-9.]+)(|/[0-9.]+)\\)";
        Pattern regex = Pattern.compile(ruleRegex);
        for(int i = 0; i < numberOfLeaves; i++) {
            if(leaves[i] != null) {

                Matcher matcher = regex.matcher(leaves[i]);
                matcher.matches();
                try {
                    String ruleText = matcher.group(1);
                    String classPredictedText = matcher.group(2);
                    String recordsInLeafText = matcher.group(3);
                    String misclassifiedText = matcher.group(4);

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

                    //set up the rule and add it to the vector
                    Rule theRule = new Rule(ruleText, accuracy, support, recordsInLeaf, ruleLength,
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
     * Take the output from a RandomForest and turn it into a RuleCollection object.
     * 
     * @param ruleStrings - output from the classifier
     * @param numRecords - number of records in the dataset
     * @param classAttr - the class attribute from the dataset
     * @return collection of rules
     */
    private RuleCollection extractRulesFromRuleStringsRandomForest(String ruleStrings, int numRecords, Attribute classAttr) {
        
        HashSet<Rule> ruleMap = new HashSet<Rule>();
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
        String ruleRegex = "^(.+) : ([a-zA-Z0-9-_!@#$%^*~'\"\\&]+) \\(([0-9.]+)(|/[0-9.]+)\\)";
        Pattern regex = Pattern.compile(ruleRegex);
        for(int i = 0; i < numberOfLeaves; i++) {
            if(leaves[i] != null) {

                Matcher matcher = regex.matcher(leaves[i]);
                matcher.matches();
                try {
                    String ruleText = matcher.group(1);
                    String classPredictedText = matcher.group(2);
                    String recordsInLeafText = matcher.group(3);
                    String misclassifiedText = matcher.group(4);

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

                    //set up the rule and add it to the vector
                    Rule theRule = new Rule(ruleText, accuracy, support, recordsInLeaf, ruleLength,
                            predictedClass, classPredictedText, classDistribution, sortType);
                    ruleMap.add(theRule);
                }
                catch(Exception e) {
                    System.out.println(leaves[i]);
                }
            }
        }
        
        
        return new RuleCollection(ruleMap);
        
    }

    /**
     * Passes the classification through to the built decision forest
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
    public static void main(String[] argv) {
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
        return "Metaclassifier implementing ForEx++: \"A New Framework for "
                + "Knowledge Discovery from Decision Forests\" for SysFor, "
                + "RandomForest and ForestPA.\n"
                + "Selects rules with a higher-than-average accuracy and "
                + "coverage and a shorter-than-average rule length.\n\n"
                + "For more information, see:\n" 
                + getTechnicalInformation().toString();
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

        String outString = "";
        if (buildStatus == BuildStatus.BS_BUILT) {
             
            StringBuilder out = new StringBuilder("");
            out.append(finalRules.toString(groupRulesViaClassValue));
            
            if(printClassifier) {
                out.append("\n").append(m_Classifier.toString());
            }
        
            outString = out.toString();
        }
        else {
            if(buildStatus == BuildStatus.BS_NOUSEFLAGS) {
                outString = "ForEx++ not built!\nSelect at least one criteria by "
                            + "which to select rules (accuracy, coverage, or rule "
                            + "length).";
            }
            else if(buildStatus == BuildStatus.BS_RANDOMFOREST_PRINT) {
                outString = "ForEx++ not built!\nRandomForest must have printClassifiers set to true (-print).";
            }
            else if(buildStatus == BuildStatus.BS_NOTCOMPATIBLE) {
                outString = "ForEx++ not built!\nWeka ForEx++ can currently only parse RandomForest, SysFor or ForestPA.";
            }
            else if(buildStatus == BuildStatus.BS_ONEATTRIBUTE) {
                outString = "ForEx++ not built!\nUse a dataset with more than one attribute.";
            }
        }
        
        return outString;

    }

    /**
     * List the possible options from the superclass
     * @return Options Enumerated
     */
    @Override
    public Enumeration<Option> listOptions() {

        Vector<Option> newVector = new Vector<Option>();
        newVector.addElement(new Option("\tWhether to print the decision forest that the "
                + "ForEx++ rules were selected from.\n"
                + "\t(default false)", "P", 0, "-P"));
        newVector.addElement(new Option("\tWhether to remove rules with no coverage"
                + "before calculating mean coverage, support, and rule length.\n"
                + "\t(default true)", "Z", 0, "-Z"));
        newVector.addElement(new Option("\tWhether to group rules by class value "
                + "in the final output.\n"
                + "\t(default true)", "GC", 0, "-GC"));
        newVector.addElement(new Option("\tSort Method for Displaying Rules.\n"
                + "\t(Default = sort by rule accuracy)",
                  "E", 1, "-E <acc | cov | len>"));
        newVector.addElement(new Option("\tWhether to use accuracy in selecting  "
                + "ForEx++ rules.\n"
                + "\t(default true)", "UA", 0, "-UA"));
        newVector.addElement(new Option("\tWhether to use coverage in selecting  "
                + "ForEx++ rules.\n"
                + "\t(default true)", "UC", 0, "-UC"));
        newVector.addElement(new Option("\tWhether to use rule length in "
                + "selecting ForEx++ rules.\n"
                + "\t(default true)", "UR", 0, "-UR"));

        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    }
    
    /**
     * Inner class for representing a rule and the stats associated with it.
     */
    private class Rule implements Serializable, Comparable {

        /** For serialization. */
        private static final long serialVersionUID = -5891208000957072995L;

        /** The text of the rule. */
        private final String ruleText;
        /** Rule accuracy */
        private final double accuracy;
        /** Rule coverage */
        private final double coverage;
        /** Actual number of records that fall in this leaf */
        private final double numRecordsInLeaf;
        /** Rule length */
        private final int length;
        /** Which class is predicted (as index to class attribute) */
        private final int predictedClass;
        /** Which class is predicted (as text) */
        private final String predictedClassLabel;
        /** Optional distribution of the classes in the leaf. */
        private final double[] classDistribution;
        /** What to compare if you sort this rule. */
        private final int sortMethod;
        
        /**
         * The constructor for the Rule.
         * @param ruleText - The text of the rule
         * @param accuracy - Rule accuracy
         * @param coverage - Rule coverage
         * @param numRecordsInLeaf - Actual number of records that fall in this leaf
         * @param length - Rule length
         * @param predictedClass - Which class is predicted (as index to class attribute)
         * @param predictedClassLabel - Which class is predicted (as text)
         * @param classDistribution - Optional distribution of the classes in the leaf
         * @param sortMethod - What to compare if you sort this rule
         */
        public Rule(String ruleText, double accuracy, double coverage, double numRecordsInLeaf, int length,
                    int predictedClass, String predictedClassLabel, double[] classDistribution,
                    int sortMethod) {
            this.ruleText = ruleText;
            this.accuracy = accuracy;
            this.coverage = coverage;
            this.numRecordsInLeaf = numRecordsInLeaf;
            this.length = length;
            this.predictedClass = predictedClass;
            this.predictedClassLabel = predictedClassLabel;
            this.classDistribution = classDistribution;
            this.sortMethod = sortMethod;
        }
        
        /**
         * Returns rule text.
         * @return rule text
         */
        public String getRuleText() {
            return ruleText;
        }
        
        /**
         * Return rule accuracy.
         * @return rule accuracy
         */
        public double getAccuracy() {
            //return Math.round(accuracy*100000.0)/100000.0;
            return accuracy;
        }
        
        /**
         * Return rule coverage
         * @return rule coverage
         */
        public double getCoverage() {
            //return Math.round(coverage*100000.0)/100000.0;
            return coverage;
        }
        
        /**
         * Return rule length
         * @return rule length
         */
        public int getLength() {
            return length;
        }
        
        /**
         * Return Records in leaf
         * @return numRecordsInLeaf
         */
        public double getNumRecordsInLeaf() {
            return numRecordsInLeaf;
        }
        
        /**
         * Return predicted class index.
         * @return predicted class index
         */
        public int getPredictedClassIndex() {
            return predictedClass;
        }
        
        /**
         * Return predicted class label.
         * @return predicted class label
         */
        public String getPredictedClassLabel() {
            return predictedClassLabel;
        }
        
        /**
         * Return class distribution for leaf.
         * @return class distribution for leaf
         */
        public double[] getClassDistribution() {
            return classDistribution;
        }
        
        /**
         * Return string representation of rule.
         * @return string representation of rule
         */
        @Override
        public String toString() {
            StringBuilder outString = new StringBuilder(ruleText).append(": ")
                    .append(predictedClassLabel);
            
            outString.append(". Confidence: ").append(String.format("%.3f", accuracy))
                    .append("; Coverage: ").append(String.format("%.3f", coverage))
                    .append(" (").append(String.format("%.0f", numRecordsInLeaf)).append(" records)")
                    .append(";");
            
            return outString.toString();
        }

        /**
         * Compare with anothre rule based on sortMethod.
         * @param o - rule to compare this to.
         * @return 1 if sort value is higher, -1 if it's lower, 0 if their the same.
         */
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
                if(coverage > other.getCoverage())
                    return 1;
                else if(coverage < other.getCoverage())
                    return -1;
                else
                    return 0;
            case SORT_LENGTH:
                if(length > other.getLength())
                    return 1;
                else if(length < other.getLength())
                    return -1;
                else
                    return 0;
            }
            
            return 0;
            
        }
        
        /**
         * Compare this rule with another.
         * @param object - the object we are comparing this Rule to.
         * @return whether this rule is the same as the other object
         */
        @Override
        public boolean equals(Object object) {         
            
            boolean equal = false;
           
            if( object instanceof Rule ) {
                 
                double dAccuracy = Math.abs(this.accuracy - ((Rule) object).getAccuracy());
                double dCoverage = Math.abs(this.coverage - ((Rule) object).getCoverage());
                if(dAccuracy <= 0.001 && dCoverage <= 0.001 && this.length == ((Rule) object).getLength()){
                    equal = true;
                }
                
            }
                        
            return equal;
            
        }
        
        /**
         * Generate a hashcode for this Rule
         * @return hashcode for this Rule
         */
        @Override
        public int hashCode() {
            int hash = 31;
            hash *= accuracy;
            hash *= length;
            hash *= numRecordsInLeaf;
            
            //test with ordered string hash
            char[] temp = ruleText.toCharArray();
            Arrays.sort(temp);
            String tempStr = new String(temp);
            hash *= tempStr.hashCode();
            
            return hash;
        }
        
    }
    
    /**
     * Inner class for a collection of rules.
     */
    private class RuleCollection implements Serializable {
         
        /** For serialization. */
        private static final long serialVersionUID = -5891208009570723995L;

        /** The set of rules. */
        private final HashSet<Rule> rules;

        /**
         * The constructor.
         * @param rules - the set of rules.
         */
        public RuleCollection(HashSet<Rule> rules) {
            this.rules = rules;
        }
        
        /**
         * Get the set of rules.
         * @return the set of rules.
         */
        public HashSet<Rule> getRules() {
            return rules;
        }
        
        /**
         * Get the mean accuracy for the rules in the set.
         * @return mean accuracy for the rules in the set
         */
        public double meanAccuracy() {
            
            double mean = 0;
            
                        
            for(Rule r : rules) {
                mean += r.getAccuracy();
            }
            
            mean /= rules.size();
            
            return Math.round(mean*100000.0)/100000.0; //round to five dec places
            
        }
        
        /**
         * Get the mean coverage for the rules in the set.
         * @return mean coverage for the rules in the set
         */
        public double meanCoverage() {
            
            double mean = 0;
            
            for(Rule r : rules) {
                mean += r.getCoverage();
            }
            
            mean /= rules.size();
            
            return Math.round(mean*100000.0)/100000.0; //round to five dec places
            
        }
        
        /**
         * Get the mean rule length for the rules in the set.
         * @return mean rule length for the rules in the set
         */
        public double meanRuleLength() {
            
            double mean = 0;
            
            for(Rule r : rules) {
                mean += r.getLength();
            }
            
            mean /= rules.size();
            
            return Math.round(mean*100000.0)/100000.0; //round to five dec places
            
        }
        
        /**
         * Get a subset of rules with accuracy greater-than or equal-to a threshold.
         * @param accuracy - threshold above or equal to which rules will be selected.
         * @return subset of rules with accuracy greater-than or equal-to "accuracy" param.
         */
        public RuleCollection getRulesGEQAccuracy(double accuracy) {
            
            HashSet<Rule> newRules = new HashSet<Rule>();
            
            for(Rule r : rules) {
                if(r.getAccuracy() >= accuracy) {
                    newRules.add(r);
                }
            }
            
            return new RuleCollection(newRules);
        }
        
        /**
         * Get a subset of rules with coverage greater-than or equal-to a threshold.
         * @param coverage - threshold above or equal to which rules will be selected.
         * @return subset of rules with coverage greater-than or equal-to "coverage" param.
         */
        public RuleCollection getRulesGEQCoverage(double coverage) {
            
            HashSet<Rule> newRules = new HashSet<Rule>();
            
            for(Rule r : rules) {
                if(r.getCoverage()>= coverage) {
                    newRules.add(r);
                }
            }
            
            return new RuleCollection(newRules);
        }
        
        /**
         * Get a subset of rules with length less-than or equal-to a threshold.
         * @param length - threshold above or equal to which rules will be selected.
         * @return subset of rules with length less-than or equal-to "length" param.
         */
        public RuleCollection getRulesLEQLength(double length) {
            
            HashSet<Rule> newRules = new HashSet<Rule>();
            
            for(Rule r : rules) {
                if(r.getLength() <= length) {
                    newRules.add(r);
                }
            }
            
            return new RuleCollection(newRules);
        }
        
        /**
         * Merge the rules from two RuleCollections and return a new RuleCollection object.
         * @param otherRules - the RuleCollection to merge with this one.
         * @return new RuleCollection with rules from this RuleCollection and otherRules.
         */
        public RuleCollection merge(RuleCollection otherRules) {
            
            HashSet<Rule> newRuleSet = new HashSet<Rule>();
            newRuleSet.addAll(rules);
            newRuleSet.addAll(otherRules.getRules());
            
            RuleCollection newRules = new RuleCollection(newRuleSet);
            
            return newRules;
            
        }
        
        /**
         * Get the subset of rules for a particular class.
         * @param classIndex - which class to extract
         * @return a RuleCollection with only rules from classIndex.
         */
        public RuleCollection getRulesByClass(int classIndex) {
            HashSet<Rule> newRules = new HashSet<Rule>();
            
            for(Rule r : rules) {
                if(r.getPredictedClassIndex() == classIndex) {
                    newRules.add(r);
                }
            }
            
            return new RuleCollection(newRules);
        }
        
        /**
         * the intersection of this rule set and another rule set. 
         * @param otherRules - the RuleCollection to intersect with this one.
         * @return the intersection of this rule set and otherRules' rule set.
         */
        public RuleCollection intersection(RuleCollection otherRules) {
            
            HashSet<Rule> newRuleSet = new HashSet<Rule>(rules);
            newRuleSet.retainAll(otherRules.getRules());
            
            return new RuleCollection(newRuleSet);
            
        }
        
        /**
         * Get a string representation of the RuleCollection with some added 
         * information on which classifier was used and how many rules were found.
         * @param group - whether to group the output by class value.
         * @return string representing the set of rules, with some added information.
         */
        public String toString(boolean group) {
            
            int numExtracted = rules.size();
            
            StringBuilder out = new StringBuilder("There were a total of ");
            out.append(totalRulesFromClassifier).append(" rules found by the ");
            out.append(m_Classifier.getClass().getName()).append(" classifier.\n");
            out.append(numExtracted).append(" ForEx++ Rules Discovered:\n\n");
            
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
                    
                    out.append("Rules for class value ").append(k).append(" (")
                            .append(classMap.get(k).size()).append(" found): \n");
                    
                    if(sortType == SORT_LENGTH) {
                        Collections.sort(classMap.get(k));
                    }
                    else
                        Collections.sort(classMap.get(k), Collections.reverseOrder());
                    
                    for (Rule rule : classMap.get(k)) {
                        
                        out.append(rule.toString()).append("\n");
                        
                    }
                    
                    out.append("\n\n");
                    
                }
                
            } //end grouping
            
            return out.toString();
        }
        
        /**
         * Return the string representation of the RuleCollection without grouping.
         * @return string representing the set of rules, with some added information.
         * @see RuleCollection#toString(boolean) 
         */
        @Override
        public String toString() {
            return toString(false);
        }
        
         
       
    }
     
    /**
     * Get whether to print the classifier
     * @return whether to print the classifier
     */
    public boolean isPrintClassifier() {
        return printClassifier;
    }

    /**
     * Set whether to print the classifier
     * @param printClassifier
     */
    public void setPrintClassifier(boolean printClassifier) {
        this.printClassifier = printClassifier;
    }
    
    /**
     * Return tip text for this option
     * @return tip text for this option
     */
    @Override
    public String classifierTipText() {
        return "Either SysFor, RandomForest or ForestPA.";
    }
    
    /**
     * Return tip text for this option
     * @return tip text for this option
     */
    public String printClassifierTipText() {
        return "Whether to print the decision forest that the ForEx++ rules were selected from.";
    }
    
    /**
     * Return whether to remove rules with no coverage before calculating mean
     * coverage, support, and rule length
     * @return whether to remove rules with no coverage before calculations
     */
    public boolean isRemoveZeroCoverageRules() {
        return removeZeroCoverageRules;
    }

    /**
     * Set whether to remove rules with no coverage before calculations
     * @param removeZeroCoverageRules
     */
    public void setRemoveZeroCoverageRules(boolean removeZeroCoverageRules) {
        this.removeZeroCoverageRules = removeZeroCoverageRules;
    }

    /**
     * Return tip text for this option
     * @return tip text for this option
     */
    public String removeZeroCoverageRulesTipText() {
        return "Whether to remove rules with no coverage before calculating mean coverage, support, and rule length.";
    }
    
    /**
     * Return whether to group rules by class value in the final output.
     * @return Whether to group rules by class value in the final output.
     */
    public boolean isGroupRulesViaClassValue() {
        return groupRulesViaClassValue;
    }

    /**
     * Set whether to group rules by class value in the final output.
     * @param groupRulesViaClassValue
     */
    public void setGroupRulesViaClassValue(boolean groupRulesViaClassValue) {
        this.groupRulesViaClassValue = groupRulesViaClassValue;
    }

    /**
     * Return tip text for this option
     * @return tip text for this option
     */
    public String groupRulesViaClassValueTipText() {
        return "Whether to group rules via their class values.";
    }
    
    /**
     * Return sort method for displaying rules
     * @return sort method for displaying rules
     */
    public SelectedTag getSortType() {
        return new SelectedTag(sortType, TAGS_SORT);
    }
    
    /**
     * Set sort method for displaying rules
     * @param newSortType
     */
    public void setSortType(SelectedTag newSortType) {
        if(newSortType.getTags() == TAGS_SORT) {
            sortType = newSortType.getSelectedTag().getID();
        }
    }
    
    /**
     * Return tip text for this option
     * @return tip text for this option
     */
    public String sortTypeTipText() {
        return "Method to sort the rules when displayed.";
    }
    
    /**
     * Return whether to use accuracy in selecting ForEx++ rules
     * @return whether to use accuracy in selecting ForEx++ rules
     */
    public boolean getUseAccuracy() {
        return useAccuracy;
    }
    
    /**
     * Return tip text for this option
     * @return tip text for this option
     */
    public String useAccuracyTipText() {
        return "Whether or not to include accuracy in the intersection of useful rules.";
    }
    
    /**
     * Set whether to use accuracy in selecting ForEx++ rules
     * @param useAccuracy
     */
    public void setUseAccuracy(boolean useAccuracy) {
        this.useAccuracy = useAccuracy;
    }
    
    /**
     * Return whether to use coverage in selecting ForEx++ rules
     * @return whether to use coverage in selecting ForEx++ rules
     */
    public boolean getUseCoverage() {
        return useCoverage;
    }

    /**
     * Set whether to use coverage in selecting ForEx++ rules
     * @param useCoverage
     */
    public void setUseCoverage(boolean useCoverage) {
        this.useCoverage = useCoverage;
    }
        
    /**
     * Return tip text for this option
     * @return tip text for this option
     */
    public String useCoverageTipText() {
        return "Whether or not to include coverage in the intersection of useful rules.";
    }
    
    /**
     * Return whether to use rule length in selecting ForEx++ rules
     * @return whether to use rule length in selecting ForEx++ rules
     */
    public boolean getUseRuleLength() {
        return useRuleLength;
    }
    
    /**
     * Set whether to use rule length in selecting ForEx++ rules
     * @param useRuleLength
     */
    public void setUseRuleLength(boolean useRuleLength) {
        this.useRuleLength = useRuleLength;
    }

    /**
     * Return tip text for this option
     * @return tip text for this option
     */
    public String useRuleLengthTipText() {
        return "Whether or not to include rule length in the intersection of useful rules.";
    }
    
}
