---
title: "Tu propio overleaf"
description: Deploya tu propio servidor de overleaf.
tags: ["open-source", "latex", "overleaf", "linux", "server"]
date: 2024-11-24T00:33:05-03:00
draft: false
author: Leonardo Pepino
authorLink: /about
math: true
---

[Overleaf](https://www.overleaf.com/) es un editor de LaTeX basado en la nube y diseñado para la colaboración. Es muy popular entre investigadores y estudiantes porque permite colaborar en la redacción de documentos utilizando LaTeX y compilarlos sin necesidad de instalar paquetes. Sin embargo, tus documentos están almacenados en servidores de terceros, y debes pagar una suscripción para colaborar con más de una persona.

En este artículo, te mostraré cómo resolví estos problemas alojando mi propia versión de Overleaf. De esta manera, no tengo que pagar suscripciones y todos los documentos permanecen en mi computadora.

### Instalando overleaf

Además del servicio en la nube, Overleaf ofrece una [edición gratuita de código abierto](https://github.com/overleaf/overleaf) que se puede instalar localmente.
La instalación es muy sencilla y está bien documentada.

1) Clonar el repositorio de Overleaf:
```bash
git clone https://github.com/overleaf/toolkit.git ./overleaf-toolkit
```
2) Instalar docker siguiendo las [instrucciones](https://docs.docker.com/engine/install/ubuntu/)
3) Hacer cd al repositorio:
```bash
cd ./overleaf-toolkit
```
4) Generar los archivos de configuración:
```bash
bin/init
```

Luego de correr el comando se crean 3 archivos en la carpeta config, los cuales pueden editarse para customizar el overleaf.
Algunas variables para customizar son:

En overleaf.rc
```
OVERLEAF_DATA_PATH=[directorio donde guardar los documentos]
OVERLEAF_PORT=[puerto para overleaf]
```

En variables.env
```
OVERLEAF_NAV_TITLE=[título para mostrar en la página]
OVERLEAF_ADMIN_EMAIL=[email para contacto]
OVERLEAF_EMAIL_SMTP_... [variables para configurar el email]
OVERLEAF_HEADER_IMAGE_URL=[logo]
```

5) Correr la instancia:
```bash
bin/up
```

6) Ir a https://127.0.0.1:[OVERLEAF_PORT]/launchpad y crear un usuario.

Ahora podes empezar a usar tu propio overleaf!

### Agregar paquetes de LaTeX

Podemos instalar todos los paquetes de TexLive, agregando unos 3-4 GB al contenedor de docker.

```bash
docker exec sharelatex tlmgr install scheme-full
docker exec sharelatex tlmgr path add
```

Luego podemos hacer commit de los cambios y guardar el contenedor actualizado:
```bash
docker commit sharelatex local/sharelatex-with-texlive-full:5.2.1
```

Hay que asegurarse de que la versión de overleaf, que se encuentra en config/version coincide con la usada en el nombre del contenedor (5.2.1 en mi caso)
Luego detiene el contenedor (haciendo Ctrl+C) y edita el archivo config/overleaf.rc actualizando la variable SHARELATEX_IMAGE_NAME:

```bash
SHARELATEX_IMAGE_NAME=local/sharelatex-with-texlive-full 
```

Lanzar nuevamente el contenedor de overleaf y chequear que anda.

### Agregar comentarios y rastrear cambios

Por defecto, no se pueden hacer comentarios en esta versión de overleaf, sin embargo, existe un [overleaf extendido](https://github.com/yu-i-i/overleaf-cep/tree/ldap-tc) que resuelve este problema.

Podemos seguir las instrucciones de este [issue](https://github.com/overleaf/overleaf/issues/1193#issuecomment-2256681075) para activar comentarios:

Corre el siguiente comando para lanzar un bash interactivo en el contenedor.

```bash
docker exec -it sharelatex bash
```
Ejecutar los siguientes comandos desde la carpeta overleaf:

```bash
git clone https://github.com/yu-i-i/overleaf-cep.git overleaf-cep
mv overleaf-cep/services/web/modules/track-changes services/web/modules/track-changes
rm -rf overleaf-cep
sed -i "/moduleImportSequence:/a 'track-changes'," services/web/config/settings.defaults.js
sed -i 's/trackChangesAvailable: false/trackChangesAvailable: true/g' services/web/app/src/Features/Project/ProjectEditorHandler.js
```

Salir de la terminal interactiva y commitear los cambios:

```bash
docker commit sharelatex local/sharelatex-with-texlive-full-tc:5.2.1
```

Finalmente, actualizar la variable SHARELATEX_IMAGE_NAME en el config/overleaf.rc

### Acceder desde afuera
Si estás contento con cómo funciona overleaf localmente, querrás acceder desde cualquier red y agregar usuarios, etc...
Hay muchas maneras de hacer eso, pero para mi la más fácil es usar CloudFlare Tunnel si se posee un dominio, o ngrok sino.

Ngrok es muy fácil de instalar:

1) Ir a https://ngrok.com/
2) Unirse o loggearse
3) Seguir las instrucciones para correr ngrok en tu computadora y agregar el token de autentificación.
4) Se nos da un dominio estático gratuito para usar. Se puede encontrar en tu dashboard de ngrok. Luego correr el comando mostrado apuntando al puerto de overleaf:
```bash
ngrok http --url=[YOUR_STATIC_DOMAIN] [OVERLEAF_PORT]
```
5) Ahora navega a tu dominio estático y disfruta de Overleaf 😃

{{< figure src="images/login-page.png" title="Enjoy!" >}}

### Sincronizar con git
Finalmente, si el servidor se muere, no queremos perder todos los datos. Se puede backupear regularmente las carpetas de overleaf, pero eso fuerza a tener que reinstalar overleaf. Otra manera es guardar directamente los documentos como si se descargaran de overleaf. Para eso, me arme un [script](https://github.com/mrpep/overleaf-git-sync) fácil de usar. Siguiendo las instrucciones del readme podrás backupear todos tus documentos también en github!
