pipeline {
    agent any          // the Jenkins master is “localhost” on Windows

    options {
        buildDiscarder(logRotator(numToKeepStr: '5'))
        timestamps()
    }

    environment {
        // Where we will collect the final artefacts
        RELEASE_DIR  = "${env.WORKSPACE}/release"
        // Make sure the script is executable (Jenkins will run it via cmd)
        BUILD_SCRIPT = "${env.WORKSPACE}/build.bat"
    }

    stages {

        /* -----------------------------------------------------------
           1.  Preparation – clone the repo and wipe any stale state
        ----------------------------------------------------------- */
        stage('Preparation') {
            steps {
                script {
                    deleteDir()      // clean workspace
                }
                checkout scm
            }
        }

        /* -----------------------------------------------------------
           2.  Build – configure CMake & generate project files
        ----------------------------------------------------------- */
        stage('Build') {
            steps {
                bat "${BUILD_SCRIPT} --clean --reldebug"
            }
        }

        /* -----------------------------------------------------------
           3.  Compile – actually compile the code (+ optional tests)
        ----------------------------------------------------------- */
        stage('Compile') {
            steps {
                bat "${BUILD_SCRIPT} --install --installdir:${RELEASE_DIR}"
            }
        }

        /* -----------------------------------------------------------
           4.  Release – archive artefacts so Jenkins keeps them
        ----------------------------------------------------------- */
        stage('Release') {
            steps {
                archiveArtifacts artifacts: "${RELEASE_DIR}/**/*", fingerprint: true
            }
        }
    }

    post {
        always {
            echo "Pipeline finished – see artefacts in ${RELEASE_DIR}"
        }
        failure {
            echo "Something went wrong – inspect the console log above"
        }
    }
}