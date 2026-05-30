[Setup]
AppName=STT Watchdog
AppVersion={#AppVersion}
AppPublisher=STT Dev
AppPublisherURL=https://github.com/zitlem/STT
DefaultDirName={autopf}\STT
DefaultGroupName=STT
OutputDir=Output
OutputBaseFilename=STT-Setup
Compression=none
SolidCompression=no
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=icon.ico

[Files]
Source: "dist\STT\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\STT Watchdog"; Filename: "{app}\STT.exe"; Parameters: "--gui"
Name: "{group}\Uninstall STT"; Filename: "{uninstallexe}"
Name: "{commondesktop}\STT Watchdog"; Filename: "{app}\STT.exe"; Parameters: "--gui"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"
Name: "autostart"; Description: "Start STT automatically at &login"; GroupDescription: "Auto-start:"; Checked: yes

[Run]
Filename: "schtasks.exe"; Parameters: "/Create /TN ""STT"" /TR ""{app}\STT.exe --headless"" /SC ONLOGON /RL HIGHEST /F"; Flags: runhidden; Tasks: autostart
; After a silent auto-update, immediately start the new binary via the scheduled task
Filename: "schtasks.exe"; Parameters: "/Run /TN ""STT"""; Flags: runhidden nowait; Tasks: autostart
; After an interactive install, offer to launch the GUI
Filename: "{app}\STT.exe"; Parameters: "--gui"; Description: "Launch STT Watchdog now"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "schtasks.exe"; Parameters: "/Delete /TN ""STT"" /F"; Flags: runhidden; RunOnceId: "RemoveTask"

[Code]
procedure KillRunningInstance();
var ResultCode: Integer;
begin
  Exec('taskkill.exe', '/F /IM STT.exe', '', SW_HIDE,
       ewWaitUntilTerminated, ResultCode);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then
    KillRunningInstance();
end;
