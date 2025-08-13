pipeline {
    // A lightweight agent for the main pipeline
    agent any

    options {
        buildDiscarder logRotator(numToKeepStr: '5')
        timestamps()
    }

    stages {
        stage('Build and Release Matrix') {
            matrix {
                axes {
                    axis {
                        name 'HOUDINI_VERSION'
                        // Adjust your target Houdini versions here
                        values '20.5.522', '20.5.584', '20.5.613'
                    }
                }

                // Each combination of axes will run on its own agent (or workspace)
                agent any

                stages {
                    // All stages below now run inside a completely isolated workspace
                    // for each Houdini version.

                    /* 1. Preparation (in the isolated workspace) */
                    stage('Preparation') {
                        steps {
                            // Clean the temporary workspace and check out the source code into it
                            deleteDir()
                            checkout scm
                        }
                    }

                    /* 2. Build (Configure) */
                    stage('Build') {
                        steps {
                            script {
                                def HFS_PATH = "C:/Program Files/Side Effects Software/Houdini ${env.HOUDINI_VERSION}"

                                // Now this works because we are in the root of our own private copy of the repo
                                bat """
                                    call build.bat --clean "--houdinipath:${HFS_PATH}"
                                """
                            }
                        }
                    }

                    /* 3. Compile (Build + Install) */
                    stage('Compile') {
                        steps {
                            script {
                                // RELEASE_DIR is now relative to the isolated workspace
                                def RELEASE_PATH = "${WORKSPACE}\\release"
                                def HFS_PATH = "C:/Program Files/Side Effects Software/Houdini ${env.HOUDINI_VERSION}"

                                bat "call build.bat --install --installdir:\"${RELEASE_PATH}\" \"--houdinipath:${HFS_PATH}\""
                            }
                        }
                    }

                    /* 4. Stash Artefacts */
                    stage('Stash') {
                        steps {
                            // Stash the release folder for later collection
                            stash name: "release-${HOUDINI_VERSION}", includes: 'release/**'
                        }
                    }
                }
            }
        }

        stage('Collect Artifacts') {
            steps {
                // Clean the main workspace and check out source to get vqvdb.json
                deleteDir()
                checkout scm

                script {
                    def versions = ['20.5.522', '20.5.584', '20.5.613']
                    def outputDir = 'vqvdb v0.0.x'  // Folder name with space
                    bat "mkdir \"${outputDir}\""

                    boolean binsCopied = false

                    for (String version : versions) {
                        unstash "release-${version}"

                        bat "mkdir \"${outputDir}\\${version}\""
                        bat "mkdir \"${outputDir}\\${version}\\dso\""
                        bat "xcopy /s /y release\\dso\\* \"${outputDir}\\${version}\\dso\\\""

                        if (!binsCopied) {
                            if (fileExists('release\\bins')) {
                                bat "mkdir \"${outputDir}\\bins\""
                                bat "xcopy /s /y release\\bins\\* \"${outputDir}\\bins\\\""
                                binsCopied = true
                            }
                        }

                        // Clean up the unstashed release folder for the next iteration
                        bat "rmdir /s /q release"
                    }

                    // Archive the organized output folder and the JSON file
                    archiveArtifacts artifacts: "${outputDir}/**/*, vqvdb.json", fingerprint: true, allowEmptyArchive: true
                }
            }
        }
    }

    post {
        always { echo 'Pipeline finished – see artefacts above' }
        failure { echo 'Pipeline failed – inspect console log' }
    }
}