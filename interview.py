
# update is a hashMap(key = field, value = hashMap(key=value=stored_info))


# user A: Kevin J Lam. -> Kevin J
# candidate: 
# location:
# email: 389@google.com



        

class update(object):

    def __init__(self, user_id):
        self.user_id = user_id
        self.deleted = dict()
        self.updated = dict()

    def delte_certain_info(self, field, info):
        
        if (not field in self.deleted.keys()):
            self.deleted[field] = {}
        self.deleted[field][info] = info

    def update_certain_info(self, field, info):
        if (not field in self.updated.keys()):
            self.updated[field] = {}
        self.updated[field][info] = info

    def revise_certain_info(self, field, before, after):
        if (not field in self.updated.keys()):
            self.updated[field] = {}

        self.updated[field][before] = after


    # info can be dict, string, list of dict
    def deleted_certain_info_by_index(self, field, info, index):
        if (not field in self.deleted.keys()):
            self.deleted[field] = {}
        
        if type(self.deleted[filed][index]) is dict:
            





    
class demenstration(object):

    def __init__(self, user_id, original, difference):
        self.user_id = user_id
        self.original = original
        self.difference = difference
        self.fields = ["email"]

    def show_content(self):
        for field in self.fields:
            original_ = self.original[field]
            updated = self.difference.updated[field]
            deleted = self.difference.deleted[field]
            self.show_individual_content(original_, updated, deleted)

    def show_individual_content(self, original, updated, deleted):
        for deleted_ in deleted:
            print(deleted_)
            del(original[deleted_])
        for updated_ in updated:
            print(updated_)
            if (updated_ in original.keys()):
                del(original[updated_])
            original[updated[updated_]] = updated[updated_]
        print(original)

        


#dict() key=field  value:list of object/string/dict
profile = dict() #default
profile["email"] = dict()
profile["email"]["mock@google.com"] = "mock@google.com"
profile["email"]["mock@gmail.com"] = "mock@gmail.com"
profile["experience"] = []
profile["experience"].append{"company":"Google", "title":"Software engineer"}
profile["experience"].append{"company":"Amazon", "title":"Manager"}




update_ = update("uid00001")
update_.delte_certain_info("email", "mock@google.com")
update_.update_certain_info("email", "mock_new@google.com")
update_.revise_certain_info("email", "mock@gmail.com", "mock_new@gmail.com")
# update_.revise_certain_info("experience", )

board = demenstration("uid00001", profile, update_)
board.show_content()

