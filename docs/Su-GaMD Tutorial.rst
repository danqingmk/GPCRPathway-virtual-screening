Su-GaMD Tutorial
================

Su-GaMD Tutorial is designed to elaborate the application of Su-GaMD technology in revealing GPCRs activation mechanisms by capturing metastable intermediate states during conformational transitions.

Based on the practical needs of selective ligand discovery (e.g., A1R antagonists), this tutorial details the entire workflow from system setup (including PDB structure processing, phospholipid bilayer construction, hydrogenation, 
and parameter file generation) to system equilibration via conventional MD (cMD), pre-GaMD simulation, and final Su-GaMD simulation. 

Prerequisites
-------------

- Operating System: Linux

- Required Software: PyMOL, Amber

- Required Tools/Websites: CHARMM-GUI, H++, charmmlipid2amber.py, box.sh, SuMD.py

Complete Workflow
-----------------

**System Setup**
^^^^^^^^^^^^^^^^

Step 1: Download PDB File from PDB Bank
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the target protein's PDB file from the PDB Bank.

Step 2: Open and Clean Structure in PyMOL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Open the downloaded PDB file in PyMOL, and remove water molecules, ions, and other unnecessary components.

2. Save the cleaned target structure in PDB format.

Step 3: Build a Phospholipid Bilayer System with CHARMM-GUI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Go to the official CHARMM-GUI website.

2. Under the "Input Generator" tab, locate "Membrane Builder" and click "Bilayer Builder".

3. Upload the unprocessed PDB file (cleaned structure without H++ treatment) and click "Next Step".

4. Choose the protein chain(s) to retain and click "Next Step".

5. Accept the default options and click "Next Step".

6. In the "Orientation Options" section, select "Run PPM 2.0" and click "Next Step".

7. Adjust the parameters to set the type and size of the membrane, then click "Next Step".

8. Select the ion types and concentrations, then click "Next Step".

9. Click "Next Step" repeatedly until the download link for the step5_assembly.pdb file appears, then download the file.

Step 4: Hydrogenation Using H++
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Go to the official H++ website.

2. Click "PROCESS A STRUCTURE".

3. Upload the cleaned PDB file from Step 2 and click "Process File".

4. Modify the parameters as needed, click "PROCESS...", and wait for the processing to complete.

5. Once complete, click "VIEW RESULTS".

6. Under "Other Useful Files," download the PDB file containing the hydrogenated structure.

Step 5: Replace Protein in Bilayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Open both the hydrogenated PDB file from Step 4 and the step5_assembly.pdb (membrane assembly file) from Step 3 in PyMOL.

2. Replace the original protein in the bilayer system with the hydrogenated protein and save it as a new PDB file.

Step 6: Modify Lipid Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the following command in the terminal to convert the lipid format (use the newly saved PDB file as input)::

    python charmmlipid2amber.py -i *.pdb -o *_out.pdb

Step 7: Calculate System's Periodic Boundary Box Size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Execute the following command to calculate the box size::

    sh box.sh -i *_out.pdb -s water

Example output (record this value for subsequent steps): 81.35599899291992 80.00199890136719 70.54999923706055

Step 8: Use Amber's tleap Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Run the following command in the terminal to enter tleap interactive mode and perform related operations::

    tleap -s -f leap.in

2. Enter the following commands in the tleap interactive interface (note to replace placeholders and the box size from Step 7)::

    complex = loadpdb *_out.pdb
    set complex box {81.35599899291992 80.00199890136719 70.54999923706055}  # Replace with actual output from Step 7
    bond complex.xx.SG complex.xx.SG  # Connect disulfide bonds; xx = actual residue number
    charge complex  # Check charge
    addions complex Na+(Cl-) 0  # Add counterions for charge neutralization
    check complex  # Check system integrity
    saveamberparm complex com.prmtop com.inpcrd  # Save parameter and coordinate files
    quit

Step 9: Generate com.inpcrd and com.prmtop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After Step 8 is completed, com.prmtop (parameter file) and com.inpcrd (coordinate file) will be generated, which will be used in subsequent equilibration and simulation steps.

**System Equilibration**
^^^^^^^^^^^^^^^^^^^^^^^^

Step 10: Equilibrate the System with cMD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Copy com.inpcrd and com.prmtop generated in Step 9 into the "3.cMD-equilibrium" folder.

2. Modify the heat.in and equil1.in files in this folder, adjust the restraint mask (restraintmask=':1-x'), where x is the last residue number of the protein.

3. Run the following command in the background to perform cMD equilibration::

    nohup sh mdrun.sh

Step 11: Generate equil2.rst for the Next Step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the cMD equilibration is completed, equil2.rst (restart file) will be generated, which will be used in the subsequent GaMD simulation.

**GaMD**
^^^^^^^^

Step 12: Run GaMD
^^^^^^^^^^^^^^^^^

1. Copy com.prmtop (generated in Step 9) and equil2.rst (generated in Step 11) into the "4.GaMD" folder.

2. Run the following command in the background to perform the GaMD simulation::

    nohup sh gamd.sh

Step 13: Generate gamd-restart.dat for the Next Step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the GaMD simulation is completed, gamd-restart.dat will be generated, which will be used in the subsequent Su-GaMD simulation.

**Su-GaMD**
^^^^^^^^^^^

Step 14: Run Su-GaMD
^^^^^^^^^^^^^^^^^^^^

1. Copy the following files into the "5.Su-GaMD" folder:

- com.prmtop (generated in Step 9)

- com.inpcrd (generated in Step 9)

- equil2.rst (generated in Step 11)

- gamd-restart.dat (generated in Step 13)

- ref.inpcrd (reference structure file, generated by tleap, which is the initial conformation of the simulation)

2. Modify the SuMD.py script (line 16) to adjust the supervision variable to either RMSD (Root Mean Square Deviation) or Distance.

3. Run the following command in the background to perform the Su-GaMD simulation::

    nohup python SuMD.py

**Notes**
^^^^^^^^^

1. All scripts (mdrun.sh, gamd.sh, SuMD.py, etc.) must have executable permissions. If permissions are insufficient, use the command "chmod +x script_name" to add them.

2. Placeholders in the steps (such as residue number xx, box size, etc.) must be replaced with actual values according to the actual experimental system.

3. The output of background commands (nohup ... &) will be saved to the nohup.out file, which can be used to view running logs and error messages.

4. Ensure compatibility between all software and tool versions. It is recommended to use the officially recommended version combination to avoid simulation failures due to version issues.




