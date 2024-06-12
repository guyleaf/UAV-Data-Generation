#Requires -Version 5.1

$blenderPath = 'F:\Program Files\Blender Foundation\bl_symlink'

$execArgs = $args

if (Test-Path -Path $blenderPath) {
    $execArgs += @("--custom-blender-path", "`"$blenderPath`"")
}
else {
    $execArgs += @("--blender-install-path", "`"$blenderPath`"")
}

Start-Process -NoNewWindow -Wait -FilePath "blenderproc" -ArgumentList $execArgs
