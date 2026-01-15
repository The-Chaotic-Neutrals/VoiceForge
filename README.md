# VoiceForge

## Servers

VoiceForge is split into multiple microservices. Each service has its own install and launch script:

- Main API/UI: `app/install/install_main.bat`, `Voice_Forge.bat`
- Chatterbox TTS: `app/install/install_chatterbox.bat`, `app/launch/launch_chatterbox_server.bat`
- Soprano TTS: `app/install/install_soprano.bat`, `app/launch/launch_soprano_server.bat`
- RVC: `app/install/install_rvc.bat`, `app/launch/launch_rvc_server.bat`
- Audio Services: `app/install/install_audio_services.bat`, `app/launch/launch_audio_services_server.bat`
- ASR: `app/install/install_asr.bat`, `app/launch/launch_asr_server.bat`

Soprano defaults to port `8894` and can be selected in the UI/extension via the **TTS Backend** dropdown.

