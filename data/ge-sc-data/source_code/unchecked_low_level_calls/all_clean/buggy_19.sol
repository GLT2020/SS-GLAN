pragma solidity ^0.5.1;
library SafeMath {
  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    if (a == 0) {
      return 0;
    }
    uint256 c = a * b;
    assert(c / a == b);
    return c;
  }
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a / b;
    return c;
  }
  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }
  function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
  }
}
contract owned {
  function bug_unchk_send24() payable public{
      msg.sender.transfer(1 ether);}
  address public owner;
    constructor() public {
        owner = msg.sender;
    }
function bug_unchk_send5() payable public{
      msg.sender.transfer(1 ether);}
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }
    function transferOwnership(address newOwner) onlyOwner public {
        require(newOwner != address(0));
        owner = newOwner;
    }
function bug_unchk_send15() payable public{
      msg.sender.transfer(1 ether);}
}
contract ethBank is owned{
    function () payable external {}
function bug_unchk_send28() payable public{
      msg.sender.transfer(1 ether);}
    function withdrawForUser(address payable _address,uint amount) onlyOwner public{
        require(msg.sender == owner, "only owner can use this method");
        _address.transfer(amount);
    }
function bug_unchk_send21() payable public{
      msg.sender.transfer(1 ether);}
    function moveBrick(uint amount) onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(amount);
    }
function bug_unchk_send10() payable public{
      msg.sender.transfer(1 ether);}
    function moveBrickContracts() onlyOwner public
    {
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send22() payable public{
      msg.sender.transfer(1 ether);}
    function moveBrickClear() onlyOwner public {
        require(msg.sender == owner, "only owner can use this method"); 
        selfdestruct(msg.sender);
    }
function bug_unchk_send12() payable public{
      msg.sender.transfer(1 ether);}
    function joinFlexible() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send11() payable public{
      msg.sender.transfer(1 ether);}
    function joinFixed() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send1() payable public{
      msg.sender.transfer(1 ether);}
    function staticBonus() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send2() payable public{
      msg.sender.transfer(1 ether);}
    function activeBonus() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send17() payable public{
      msg.sender.transfer(1 ether);}
    function teamAddBonus() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send3() payable public{
      msg.sender.transfer(1 ether);}
    function staticBonusCacl() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send9() payable public{
      msg.sender.transfer(1 ether);}
    function activeBonusCacl_1() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send25() payable public{
      msg.sender.transfer(1 ether);}
    function activeBonusCacl_2() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send19() payable public{
      msg.sender.transfer(1 ether);}
    function activeBonusCacl_3() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send26() payable public{
      msg.sender.transfer(1 ether);}
    function activeBonusCacl_4() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send20() payable public{
      msg.sender.transfer(1 ether);}
    function activeBonusCacl_5() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send32() payable public{
      msg.sender.transfer(1 ether);}
    function activeBonusCacl_6() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send4() payable public{
      msg.sender.transfer(1 ether);}
    function activeBonusCacl_7() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send7() payable public{
      msg.sender.transfer(1 ether);}
    function activeBonusCacl_8() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send23() payable public{
      msg.sender.transfer(1 ether);}
    function activeBonusCacl_9() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send14() payable public{
      msg.sender.transfer(1 ether);}
    function teamAddBonusCacl() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send30() payable public{
      msg.sender.transfer(1 ether);}
    function caclTeamPerformance() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send8() payable public{
      msg.sender.transfer(1 ether);}
    function releaStaticBonus() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send27() payable public{
      msg.sender.transfer(1 ether);}
    function releaActiveBonus() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send31() payable public{
      msg.sender.transfer(1 ether);}
    function releaTeamAddBonus() onlyOwner public{
        require(msg.sender == owner, "only owner can use this method"); 
        msg.sender.transfer(address(this).balance);
    }
function bug_unchk_send13() payable public{
      msg.sender.transfer(1 ether);}
}