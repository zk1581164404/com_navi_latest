sudo add-apt-repository ppa:git-core/ppa
sudo apt-get update
sudo apt-get install git

ubuntu16.4以及以下版本需要更新git

refuse connection:443 的问题 使用下面方法：
git config --global url."https://ghproxy.com/https://github.com".insteadOf "https://github.com"
