[app]

# (str) Title of your application
title = Analisador de Audio

# (str) Package name
package.name = analisadoraudio

# (str) Package domain (needed for android/ios packaging)
package.domain = com.henriquealmeida.analisadoraudio

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas,wav,mp3

# (str) Application versioning (method 1)
version = 1.0

# (list) Application requirements
# comma separated e.g. requirements = sqlite3,kivy
requirements = python3,kivy,numpy,scipy,matplotlib,soundfile,pandas,garden.matplotlib

# (str) Supported orientation (landscape, portrait or all)
orientation = portrait

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 0

# (list) Permissions
android.permissions = WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE,RECORD_AUDIO,INTERNET

# (int) Target Android API, should be as high as possible.
android.api = 30

# (int) Minimum API your APK will support.
android.minapi = 21

# (str) Android NDK version to use
android.ndk = 23b

# (str) Android SDK version to use
android.sdk = 30

# (bool) Use --private data storage (True) or --dir public storage (False)
android.private_storage = True

# (str) Android app theme, default is ok for Kivy-based app
android.theme = "@android:style/Theme.NoTitleBar"

# (list) Java classes to add as java src in the apk
android.add_src = 

# (str) OUYA Console category. Should be one of GAME or APP
android.ouya.category = APP

# (str) Filename of OUYA Console icon. It must be a 732x412 png image.
android.ouya.icon.filename = %(source.dir)s/data/ouya_icon.png

# (str) XML file to include as an intent filters in <activity> tag
android.manifest.intent_filters = 

# (str) launchMode to set for the main activity
android.manifest.launch_mode = standard

# (list) Android additional libraries to copy into libs/armeabi
android.add_jars = 

# (list) Android additional libraries to copy into libs/armeabi-v7a
android.add_aars = 

# (list) put these files or directories in the apk assets directory.
# Either form may be used, and assets need not be in 'source.include_exts'.
android.add_assets = 

# (list) put these files or directories in the apk res directory.
# The option may be used in three ways, the value may contain one or zero ':'
# Some examples:
# 1) A file to place in the 'drawable' resource directory:
#    android.add_resources = images/image.png:drawable
# 2) A directory, here 'images' must contain resources in one or more res subdirectories:
#    android.add_resources = images:res
# 3) A directory to be put in the res directory. The final resource location would be:
#    android.add_resources = %(source.dir)s/data/images:images
android.add_resources = 

# (str) Name of the certificate to use for signing the debug version
android.debug_keystore = ~/.android/debug.keystore

# (str) Name of the certificate to use for signing the release version
android.release_keystore = %(source.dir)s/my-release-key.keystore

# (str) Path to a custom whitelist file
android.whitelist = 

# (str) Path to a custom blacklist file
android.blacklist = 

[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = False, 1 = True)
warn_on_root = 1

# (str) Path to build artifact storage, absolute or relative to spec file
build_dir = ./.buildozer

# (str) Path to build output (i.e. .apk, .ipa) storage
bin_dir = ./bin
