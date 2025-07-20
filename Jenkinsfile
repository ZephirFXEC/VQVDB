pipeline {
	agent any
    options {
		buildDiscarder logRotator(numToKeepStr: '5')
        timestamps()
    }

    environment {
	// where the artefacts will be collected
        RELEASE_DIR = "${WORKSPACE}\\release"
    }

    stages {

		/* 1. Clone the repo and wipe any leftovers */
        stage('Preparation') {
			steps {
				deleteDir()
                checkout scm
            }
        }

        /* 3. Build (CMake configure) */
		stage('Build') {
			steps {
				bat """
					call build.bat --clean
				"""
			}
		}

        /* 4. Compile (build + install) */
        stage('Compile') {
			steps {
				bat "call build.bat --install --installdir:\"${RELEASE_DIR}\""
            }
        }

        /* 5. Archive artefacts */
        stage('Release') {
			steps {
				archiveArtifacts artifacts: "${RELEASE_DIR}\\**\\*", fingerprint: true
            }
        }
    }

    post {
		always  { echo 'Pipeline finished – see artefacts above' }
        failure { echo 'Pipeline failed – inspect console log' }
    }
}
