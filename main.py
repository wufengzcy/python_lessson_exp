"""程序入口：初始化数据库 -> 登录 -> 主窗口"""

import db
from ui.login_window import LoginWindow


def main():
    db.init_db()

    def on_login_success(user):
        from ui.main_window import MainWindow

        app = MainWindow(user)
        app.mainloop()

    login = LoginWindow(on_login_success)
    login.mainloop()


if __name__ == "__main__":
    main()
