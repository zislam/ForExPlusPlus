/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Copyright (C) 2017 Charles Sturt University
 */

package weka.classifiers.meta;

import weka.classifiers.AbstractClassifierTest;
import weka.classifiers.Classifier;

import junit.framework.Test;
import junit.framework.TestSuite;
import weka.classifiers.meta.ForExPlusPlus;

/**
 * Tests ForExPlusPlusTest. Run from the command line with:<p/>
 * java weka.classifiers.trees.ForExPlusPlusTest
 *
 * @author Michael Furner (mfurner at csu dot edu dot au)
 * @version 1.0
 */
public class ForExPlusPlusTest 
  extends AbstractClassifierTest {

  public ForExPlusPlusTest(String name) { 
    super(name);  
  }

  /** Creates a default RandomTree
   * @return  SysFor*/
  public Classifier getClassifier() {
    return new ForExPlusPlus();
  }

  public static Test suite() {
    return new TestSuite(ForExPlusPlusTest.class);
  }

  public static void main(String[] args){
    junit.textui.TestRunner.run(suite());
  }
}
