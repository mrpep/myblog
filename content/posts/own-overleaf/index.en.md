---
title: "Your own overleaf"
description: Deploy your own overleaf server with comments enabled and all latex packages installed.
tags: ["open-source", "latex", "overleaf", "linux", "server"]
date: 2024-11-24T00:33:05-03:00
draft: false
author: Leonardo Pepino
authorLink: /about
math: true
---

[Overleaf](https://www.overleaf.com/) is a collaborative cloud-based latex editor. It's very popular among researchers and students as it allows to collaborate in document writing using Latex, and to compile these documents without having to install packages. However, your documents are in somebody else server, and you have to pay a subscription to collaborate with more than one person.

In this article, I'm going to show you how I overcame these problems by hosting my own version of overleaf. This way, I don't have to pay subscriptions, and all the documents stay in my computer.

### Running the official overleaf

Apart from offering the cloud service, Overleaf has released a [free community edition](https://github.com/overleaf/overleaf) that can be installed locally.

Installation is very easy and well documented.

1) Clone the Overleaf toolkit repository:
```bash
git clone https://github.com/overleaf/toolkit.git ./overleaf-toolkit
```
2) Install docker following their [instructions](https://docs.docker.com/engine/install/ubuntu/)
3) Move to the repository:
```bash
cd ./overleaf-toolkit
```
4) And generate the config files:
```bash
bin/init
```

After running the command you'll find 3 files in the config folder, which can be edited to customize the Overleaf instance.
Some things you might want to tweak are:

In overleaf.rc
```
OVERLEAF_DATA_PATH=[where you want documents to be stored]
OVERLEAF_PORT=[port for overleaf]
```

In variables.env
```
OVERLEAF_NAV_TITLE=[your cool title to display]
OVERLEAF_ADMIN_EMAIL=[who to contact when things stop working]
OVERLEAF_EMAIL_SMTP_... [these variables are to setup the email server]
OVERLEAF_HEADER_IMAGE_URL=[your cool logo]
```

5) Launch it:
```bash
bin/up
```

6) Go to https://127.0.0.1:[OVERLEAF_PORT]/launchpad and create your user.

Now you can start using your own Overleaf!

### Adding packages

We can install all the available TexLive packages, adding between 3 and 4 GB to the container.

```bash
docker exec sharelatex tlmgr install scheme-full
docker exec sharelatex tlmgr path add
```

After that we can commit the changes and save the updated container:
```bash
docker commit sharelatex local/sharelatex-with-texlive-full:5.2.1
```

Make sure that the overleaf version, found in config/version, matches the container name (5.2.1 in my case).
Then stop the overleaf container (by running Ctrl+C), and edit the config/overleaf.rc file updating the SHARELATEX_IMAGE_NAME variable:

```bash
SHARELATEX_IMAGE_NAME=local/sharelatex-with-texlive-full 
```

Launch again the overleaf container and check that it works.

### Adding comments and tracking changes

By default the overleaf community edition image doesn't have the comment feature.
Luckily, some smart people made an [extended overleaf](https://github.com/yu-i-i/overleaf-cep/tree/ldap-tc) tackling this issue.
We can follow the instructions in this [issue](https://github.com/overleaf/overleaf/issues/1193#issuecomment-2256681075) to enable comments.

Run the following command to open an interactive bash terminal in the overleaf container:

```bash
docker exec -it sharelatex bash
```
Then make sure you are in the overleaf folder and execute these commands:

```bash
git clone https://github.com/yu-i-i/overleaf-cep.git overleaf-cep
mv overleaf-cep/services/web/modules/track-changes services/web/modules/track-changes
rm -rf overleaf-cep
sed -i "/moduleImportSequence:/a 'track-changes'," services/web/config/settings.defaults.js
sed -i 's/trackChangesAvailable: false/trackChangesAvailable: true/g' services/web/app/src/Features/Project/ProjectEditorHandler.js
```

Exit the interactive terminal and then commit the changes to the image:
```bash
docker commit sharelatex local/sharelatex-with-texlive-full-tc:5.2.1
```

Finally, update the SHARELATEX_IMAGE_NAME variable in the config/overleaf.rc file.

### Access from the outside world
If you are happy with how overleaf works locally, you might want to access it from any network, and add users, etc...
There are many ways to do that, but I find it very simple to use CloudFlare Tunnel if you own a domain, or ngrok if you don't.

Ngrok is very easy to setup:

1) Go to https://ngrok.com/
2) Sign up or Login
3) Follow the instructions to run ngrok in your computer and setup the auth token.
4) You get a free static domain to use. It will be shown in your dashboard. Run the command shown and change the port to the overleaf one:
```bash
ngrok http --url=[YOUR_STATIC_DOMAIN] [OVERLEAF_PORT]
```
5) Now go to your domain url and enjoy your free overleaf 😃

{{< figure src="images/login-page.png" title="Enjoy!" >}}

### Sync with git
Finally, if your server dies, you don't want to lose all those precious latex files. You can regularly backup the overleaf folders, which is fine, but it kind of forces you to reinstall the overleaf container again and hope everything still works.

Another approach is to have a github repository where the tex files are backuped. For that purpose I created this [script](https://github.com/mrpep/overleaf-git-sync), which is very easy to use. You can check the readme with instructions, but basically all is needed is to create a repository, and then modify the config.json file adding details about the domain you used in the previous step, username, password, and the local path to the repository you created for backup.

### Associate an email
You might want to invite new users or let them reset their passwords. The manual way to do this is to go to Admin/Manage Users and register the user email. This will generate an invitation link that you can share with the person. Remember to replace localhost with your overleaf domain from Ngrok.

However this can be tedious as you have to manually do this for every new user. A better approach is to associate an email for administration purposes. I will explain the process for gmail accounts:

1) Generate an app password. This is needed as gmail will ask for 2 factor authentication, but we need a password that overleaf can use to access our account. Go [here](https://myaccount.google.com/apppasswords), enter an app name, ie. Overleaf, and then copy the generated password.
2) Modify these settings in the variables.env file:
```
OVERLEAF_EMAIL_FROM_ADDRESS=your-username@gmail.com
OVERLEAF_EMAIL_SMTP_HOST=smtp.gmail.com
OVERLEAF_EMAIL_SMTP_PORT=587
OVERLEAF_EMAIL_SMTP_SECURE=false
OVERLEAF_EMAIL_SMTP_USER=your-username@gmail.com
OVERLEAF_EMAIL_SMTP_PASS=the-pass-you-created
```
3) Run bin/up again to refresh the instance and we are ready to go!

