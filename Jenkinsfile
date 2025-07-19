pipeline {
	agent any
    options {
		buildDiscarder logRotator(numToKeepStr: '5')
        timestamps()
    }

    environment {
		// where the artefacts will be collected
        RELEASE_DIR = "${WORKSPACE}\\release"

        // LibTorch will be unpacked to WORKSPACE\libtorch
        TORCH_URL = 'https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.1.2%%2Bcu118.zip'
        TORCH_ZIP = 'libtorch.zip'
    }

    stages {

		/* 1. Clone the repo and wipe any leftovers */
        stage('Preparation') {
			steps {
				deleteDir()
                checkout scm
            }
        }

        /* 2. Download & unpack LibTorch ONLY if missing */
        stage('Download LibTorch') {
			steps {
				script {
					if (!fileExists('libtorch\\share\\cmake\\Torch\\TorchConfig.cmake')) {
						bat """
                            echo *** Downloading LibTorch ...
                            curl -L -o ${TORCH_ZIP} ^ ${TORCH_URL}
                            powershell -NoP -Command "Expand-Archive -Path libtorch.zip -DestinationPath . -Force"
                        """
                    } else {
						echo "*** LibTorch already present – skipping download."
                    }
                }
            }
        }

        /* 3. Build (CMake configure) */
        stage('Build') {
			steps {
				bat 'call build.bat --clean --reldebug'
            }
        }

        /* 4. Compile (build + tests + install) */
        stage('Compile') {
			steps {
				bat "call build.bat --reldebug --tests --install --installdir:\"${RELEASE_DIR}\""
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