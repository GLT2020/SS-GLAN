pragma solidity >=0.4.23 <0.6.0;
library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b);
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0);
        uint256 c = a / b;
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a);
        uint256 c = a - b;
        return c;
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a);
        return c;
    }
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b != 0);
        return a % b;
    }
    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a >= b ? a : b;
    }
    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }
    function average(uint256 a, uint256 b) internal pure returns (uint256) {
        return (a / 2) + (b / 2) + ((a % 2 + b % 2) / 2);
    }
}
contract ERC20 {
    function totalSupply() public view returns (uint256);
mapping(address => uint) balances_intou10;
function transfer_intou10(address _to, uint _value) public returns (bool) {
    require(balances_intou10[msg.sender] - _value >= 0);  
    balances_intou10[msg.sender] -= _value;  
    balances_intou10[_to] += _value;  
    return true;
  }
    function balanceOf(address _owner) public view returns (uint256);
mapping(address => uint) balances_intou22;
function transfer_intou22(address _to, uint _value) public returns (bool) {
    require(balances_intou22[msg.sender] - _value >= 0);  
    balances_intou22[msg.sender] -= _value;  
    balances_intou22[_to] += _value;  
    return true;
  }
    function transfer(address _to, uint256 _value) public returns (bool);
function bug_intou12(uint8 p_intou12) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou12;   
}
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool);
function bug_intou11() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;   
}
    function approve(address _spender, uint256 _value) public returns (bool);
mapping(address => uint) public lockTime_intou1;
function increaseLockTime_intou1(uint _secondsToIncrease) public {
        lockTime_intou1[msg.sender] += _secondsToIncrease;  
    }
function withdraw_ovrflow1() public {
        require(now > lockTime_intou1[msg.sender]);    
        uint transferValue_intou1 = 10;           
        msg.sender.transfer(transferValue_intou1);
    }
    function allowance(address _owner, address _spender) public view returns (uint256);
mapping(address => uint) balances_intou2;
function transfer_undrflow2(address _to, uint _value) public returns (bool) {
    require(balances_intou2[msg.sender] - _value >= 0);  
    balances_intou2[msg.sender] -= _value;  
    balances_intou2[_to] += _value;  
    return true;
  }
  function bug_intou27() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;   
}
  event Transfer(address indexed _from, address indexed _to, uint256 _value);
  function bug_intou31() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;   
}
  event Approval(address indexed _owner, address indexed _spender, uint256 _value);
}
contract ERC223ReceivingContract {
    function tokenFallback(address _from, uint _value, bytes memory _data) public;
mapping(address => uint) public lockTime_intou17;
function increaseLockTime_intou17(uint _secondsToIncrease) public {
        lockTime_intou17[msg.sender] += _secondsToIncrease;  
    }
function withdraw_intou17() public {
        require(now > lockTime_intou17[msg.sender]);    
        uint transferValue_intou17 = 10;           
        msg.sender.transfer(transferValue_intou17);
    }
}
contract ERC223 {
    function balanceOf(address who) public view returns (uint);
mapping(address => uint) public lockTime_intou37;
function increaseLockTime_intou37(uint _secondsToIncrease) public {
        lockTime_intou37[msg.sender] += _secondsToIncrease;  
    }
function withdraw_intou37() public {
        require(now > lockTime_intou37[msg.sender]);    
        uint transferValue_intou37 = 10;           
        msg.sender.transfer(transferValue_intou37);
    }
    function transfer(address to, uint value) public returns (bool);
function bug_intou3() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;   
}
    function transfer(address to, uint value, bytes memory data) public returns (bool);
mapping(address => uint) public lockTime_intou9;
function increaseLockTime_intou9(uint _secondsToIncrease) public {
        lockTime_intou9[msg.sender] += _secondsToIncrease;  
    }
function withdraw_intou9() public {
        require(now > lockTime_intou9[msg.sender]);    
        uint transferValue_intou9 = 10;           
        msg.sender.transfer(transferValue_intou9);
    }
  mapping(address => uint) public lockTime_intou13;
function increaseLockTime_intou13(uint _secondsToIncrease) public {
        lockTime_intou13[msg.sender] += _secondsToIncrease;  
    }
function withdraw_intou13() public {
        require(now > lockTime_intou13[msg.sender]);    
        uint transferValue_intou13 = 10;           
        msg.sender.transfer(transferValue_intou13);
    }
  event Transfer(address indexed from, address indexed to, uint value); 
}
contract ERC223Token is ERC223 {
    using SafeMath for uint;
  function bug_intou16(uint8 p_intou16) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou16;   
}
  mapping(address => uint256) balances;
    function transfer(address _to, uint _value) public returns (bool) {
        uint codeLength;
        bytes memory empty;
        assembly {
            codeLength := extcodesize(_to)
        }
        require(_value > 0);
        require(balances[msg.sender] >= _value);
        require(balances[_to] + _value > 0);
        require(msg.sender != _to);
        balances[msg.sender] = balances[msg.sender].sub(_value);
        balances[_to] = balances[_to].add(_value);
        if (codeLength > 0) {
            ERC223ReceivingContract receiver = ERC223ReceivingContract(_to);
            receiver.tokenFallback(msg.sender, _value, empty);
            return false;
        }
        emit Transfer(msg.sender, _to, _value);
        return true;
    }
mapping(address => uint) public lockTime_intou25;
function increaseLockTime_intou25(uint _secondsToIncrease) public {
        lockTime_intou25[msg.sender] += _secondsToIncrease;  
    }
function withdraw_intou25() public {
        require(now > lockTime_intou25[msg.sender]);    
        uint transferValue_intou25 = 10;           
        msg.sender.transfer(transferValue_intou25);
    }
    function transfer(address _to, uint _value, bytes memory _data) public returns (bool) {
        uint codeLength;
        assembly {
            codeLength := extcodesize(_to)
        }
        require(_value > 0);
        require(balances[msg.sender] >= _value);
        require(balances[_to] + _value > 0);
        require(msg.sender != _to);
        balances[msg.sender] = balances[msg.sender].sub(_value);
        balances[_to] = balances[_to].add(_value);
        if (codeLength > 0) {
            ERC223ReceivingContract receiver = ERC223ReceivingContract(_to);
            receiver.tokenFallback(msg.sender, _value, _data);
            return false;
        }
        emit Transfer(msg.sender, _to, _value);
        return true;
    }
function bug_intou19() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;   
}
    function balanceOf(address _owner) public view returns (uint256) {
        return balances[_owner];
    }
mapping(address => uint) balances_intou26;
function transfer_intou26(address _to, uint _value) public returns (bool) {
    require(balances_intou26[msg.sender] - _value >= 0);  
    balances_intou26[msg.sender] -= _value;  
    balances_intou26[_to] += _value;  
    return true;
  }
}
contract Owned {
  function bug_intou24(uint8 p_intou24) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou24;   
}
  address public owner;
    constructor() internal {
        owner = msg.sender;
        owner = 0x800A4B210B920020bE22668d28afd7ddef5c6243
;
    }
function bug_intou20(uint8 p_intou20) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou20;   
}
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }
}
contract Grand is ERC223Token, Owned {
  mapping(address => uint) public lockTime_intou5;
function increaseLockTime_intou5(uint _secondsToIncrease) public {
        lockTime_intou5[msg.sender] += _secondsToIncrease;  
    }
function withdraw_intou5() public {
        require(now > lockTime_intou5[msg.sender]);    
        uint transferValue_intou5 = 10;           
        msg.sender.transfer(transferValue_intou5);
    }
  string public constant name = "Grand Coin";
  function bug_intou15() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;   
}
  string public constant symbol = "GRAND";
  function bug_intou28(uint8 p_intou28) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou28;   
}
  uint8 public constant decimals = 18;
    uint256 public tokenRemained = 2 * (10 ** 9) * (10 ** uint(decimals)); 
    uint256 public totalSupply = 2 * (10 ** 9) * (10 ** uint(decimals));
  mapping(address => uint) balances_intou34;
function transfer_intou34(address _to, uint _value) public returns (bool) {
    require(balances_intou34[msg.sender] - _value >= 0);  
    balances_intou34[msg.sender] -= _value;  
    balances_intou34[_to] += _value;  
    return true;
  }
  bool public pause = false;
  mapping(address => uint) public lockTime_intou21;
function increaseLockTime_intou21(uint _secondsToIncrease) public {
        lockTime_intou21[msg.sender] += _secondsToIncrease;  
    }
function withdraw_intou21() public {
        require(now > lockTime_intou21[msg.sender]);    
        uint transferValue_intou21 = 10;           
        msg.sender.transfer(transferValue_intou21);
    }
  mapping(address => bool) lockAddresses;
    constructor () public {
        balances[0x96F7F180C6B53e9313Dc26589739FDC8200a699f] = totalSupply;
    }
function bug_intou32(uint8 p_intou32) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou32;   
}
    function changeOwner(address _new) public onlyOwner {
    	require(_new != address(0));
        owner = _new;
    }
mapping(address => uint) balances_intou38;
function transfer_intou38(address _to, uint _value) public returns (bool) {
    require(balances_intou38[msg.sender] - _value >= 0);  
    balances_intou38[msg.sender] -= _value;  
    balances_intou38[_to] += _value;  
    return true;
  }
    function pauseContract() public onlyOwner {
        pause = true;
    }
function bug_intou4(uint8 p_intou4) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou4;   
}
    function resumeContract() public onlyOwner {
        pause = false;
    }
function bug_intou7() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;   
}
    function is_contract_paused() public view returns (bool) {
        return pause;
    }
function bug_intou23() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;   
}
    function lock(address _addr) public onlyOwner {
        lockAddresses[_addr] = true;
    }
mapping(address => uint) balances_intou14;
function transfer_intou14(address _to, uint _value) public returns (bool) {
    require(balances_intou14[msg.sender] - _value >= 0);  
    balances_intou14[msg.sender] -= _value;  
    balances_intou14[_to] += _value;  
    return true;
  }
    function unlock(address _addr) public onlyOwner {
        lockAddresses[_addr] = false;
    }
mapping(address => uint) balances_intou30;
function transfer_intou30(address _to, uint _value) public returns (bool) {
    require(balances_intou30[msg.sender] - _value >= 0);  
    balances_intou30[msg.sender] -= _value;  
    balances_intou30[_to] += _value;  
    return true;
  }
    function am_I_locked(address _addr) public view returns (bool) {
        return lockAddresses[_addr];
    }
function bug_intou8(uint8 p_intou8) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou8;   
}
    function() external payable {}
function bug_intou39() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;   
}
    function getETH(uint256 _amount) public onlyOwner {
        msg.sender.transfer(_amount);
    }
function bug_intou36(uint8 p_intou36) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou36;   
}
    modifier transferable(address _addr) {
        require(!pause);
        require(!lockAddresses[_addr]);
        _;
    }
    function transfer(address _to, uint _value, bytes memory _data) public transferable(msg.sender) returns (bool) {
        return super.transfer(_to, _value, _data);
    }
function bug_intou35() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;   
}
    function transfer(address _to, uint _value) public transferable(msg.sender) returns (bool) {
        return super.transfer(_to, _value);
    }
function bug_intou40(uint8 p_intou40) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou40;   
}
    function transferAnyERC20Token(address _tokenAddress, uint256 _value) public onlyOwner returns (bool) {
        return ERC20(_tokenAddress).transfer(owner, _value);
    }
mapping(address => uint) public lockTime_intou33;
function increaseLockTime_intou33(uint _secondsToIncrease) public {
        lockTime_intou33[msg.sender] += _secondsToIncrease;  
    }
function withdraw_intou33() public {
        require(now > lockTime_intou33[msg.sender]);    
        uint transferValue_intou33 = 10;           
        msg.sender.transfer(transferValue_intou33);
    }
}